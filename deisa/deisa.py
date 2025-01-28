###################################################################################################
# Copyright (c) 2020-2022 Centre national de la recherche scientifique (CNRS)
# Copyright (c) 2020-2022 Commissariat a l'énergie atomique et aux énergies alternatives (CEA)
# Copyright (c) 2020-2022 Institut national de recherche en informatique et en automatique (Inria)
# Copyright (c) 2020-2022 Université Paris-Saclay
# Copyright (c) 2020-2022 Université de Versailles Saint-Quentin-en-Yvelines
#
# SPDX-License-Identifier: MIT
#
###################################################################################################

import numpy as np
from numpy.typing import NDArray, DTypeLike

# dask
from dask.array import Array  # type: ignore
from dask.distributed import Client, Queue, Variable, Worker
from dask.distributed import wait
from dask.distributed import worker_client
from dask.highlevelgraph import HighLevelGraph
import dask
import dask.array as da
from collections import namedtuple, defaultdict
from typing import NewType, Optional
import itertools
import json
import os


# Dask related
WORKERS_NAME = "workers"
ARRAYS_NAME = "arrays"

# Bridge and lock related
BRIDGE_LOCK_NAME = "nb-bridges-lock"
NB_BRIDGES_NAME = "nb-bridges"

# Contract variable name
CONTRACT_NAME = "contract"

# Name of dict keys shared by PDI
SIZE_NAME = "sizes"
SUBSIZE_NAME = "subsizes"
DTYPE_NAME = "dtype"
TIME_DIMENSION_NAME = "timedim"
START_NAME = "starts"

# In the future: before deleting, using python native slice type (since its the same), results
# in Dask complaining it is not msgPack-encodable. Namedtuple gets past this.
MySlice = namedtuple("MySlice", ["start", "stop", "step"])

# Type for full specification of dimensions - a list of MySlice. The list must have len==dims of
# the array being shared by PDI. So a 3D array (including time dimension) corresponds to a list of
# 3 MySlice instances.
DimsSpec = NewType("DimsSpec", list[MySlice])

# A contract is a dictionary of key-value pairs where the keys are the names of the array being
# shared by PDI and the values are an instance of DimsSpec.
# Ex:
# {
# "global_t": [MySlice(0,200,1), MySlice(20,100,2)]
# "global_f": [MySlice(10,500,1), MySlice(0,10,1)]
# }
ContractType = NewType("ContractType", dict[str, DimsSpec])


class ContractError(Exception):
    """
    Error class designed to alert user that the contract has been mishandled.
    """

    def __init__(self, message):
        super().__init__(message)

    def __str__(self) -> str:
        return super().__str__()


def create_client_connected_to_scheduler_at(scheduler_address: str) -> Client:
    """
    Create a client and connect to a Dask scheduler at a given address.

    Parameters
    ----------
        - scheduler_address: the address of the Dask scheduler

    Output
    ----------
        - A Dask client which is connected to the Dask scheduler.
    """
    try:
        client = Client(scheduler_address)
        return client
    except Exception as e:
        print(
            f"Failed to create a client connected to scheduler at {scheduler_address}"
            f"because of: \n{e}\nRetrying connection...\n"
        )
        return create_client_connected_to_scheduler_at(scheduler_address)


def get_bridge_instance(
    sched_file: list[int],
    mpi_rank: int,
    mpi_size: int,
    arrays_description: dict[str, dict],
    arrays_description_dtype: dict[str, DTypeLike],
    **kwargs,
):
    """
    Helper function to instantiate Bridge class from PDI.

    Parameters
    ----------
        - sched_file: A list of character encodings for the file name of the scheduler
        configuration file.
        - mpi_rank: the mpi_rank of this Bridge.
        - mpi_size: the total size of the MPI processes.
        - arrays_description: a dictionary where keys are the name of the arrays to be shared and the
        keys are dictionaries which describe each array that will be shared.
        For instance when sharing an array called "global_t", the dictionary might
        look like this:
            arrays_description = {
                'global_t': {
                    'timedim': [0],
                    'subsizes': [1, 10, 20],
                    'starts': [0, 0, 0], # varying per MPI process!
                    'sizes': [10, 20, 20]
                }
                'global_p': {
                    'timedim': [0],
                    'subsizes': [1, 20, 100],
                    'starts': [0, 0, 0], # varying per MPI process!
                    'sizes': [30, 40, 100]
                }
            }
        - arrays_description_dtype: a dictionary where keys are the name of the array and values are
        dtype of the underlying data being shared.
        For example:
            deisa_arrays_dtype = {
                'global_t': double
                'global_p': double
                }

    Output
    ----------
        - A Bridge instance
    """
    return BridgeV1(
        scheduler_encoding=sched_file,
        mpi_rank=mpi_rank,
        mpi_size=mpi_size,
        arrays_description=arrays_description,
        arrays_description_dtype=arrays_description_dtype,
        **kwargs,
    )


class DeisaArray:
    """
    Class which contains the name of the dask array being shared and the dask array itself.
    It is responsible for setting the contract that will specify which data we want in each
    dimension.

    The dask array it contains is a "global" view of the data shared by all MPI
    processes. In other words, if 4 MPI processes are each sharing a (100, 100) array
    (which are subparts of a grid divided in 2x2) at each timestep, for 5 timesteps, the
    corresponding DeisaArray will have a dask array which will have shape (5, 200, 200).
    """

    def __init__(self, name: str, array: Array):
        """
        Initialize a DeisaArray object.

        Parameters
        ----------
            - name: the name of the array.
            - array: the Dask array we are sharing.
            - selection: a list of slices per dimension of the array. It is used to select the data
            that the user needs in each dimension. In this way, PDI knows that to share and what to
            avoid sharing.
        """
        self.name = name
        self.array = array
        # default selection is None, i.e. we dont need the data.
        # TODO currently contracts are not supported, so setting this doesn't do anything.
        self.selection: Optional[DimsSpec] = None

    def normalize_slice(
        self,
        slice_start: int | None,
        slice_end: int | None,
        slice_step: int | None,
        dim: int,
    ) -> MySlice:
        """
        Applies slicing rules along a specific axis/index of the array.

        Parameters
        ----------
            - slice_start: starting index of the slice.
            - slice_end: end index of the slice.
            - slice_step: step of the slice.
            - dim: dimension/axis over which slicing occurs.

        Output
        ----------
            - A MySlice object i.e a namedtuple[int,int,int] which represents the slice start, end,
            and step in the specified dimension.
        """
        shape: tuple[int] = self.array.shape

        if slice_start is None:
            slice_start = 0
        elif slice_start < 0:
            slice_start = shape[dim] + slice_start

        if slice_end is None:
            slice_end = shape[dim]
        elif slice_end < 0:
            slice_end = shape[dim] + slice_end

        if slice_step is None:
            slice_step = 1
        elif slice_step < 0:
            raise ValueError(f"{slice_step} only positive step values are accepted")

        return MySlice(slice_start, slice_end, slice_step)

    def __getitem__(self, idx: tuple) -> Array:
        """
        Support basic dask syntax for slicing and sets the selection variable which is used
        to generate a contract.

        Parameters
        ----------
            - idx: tuple of mix of slice or int or ellipsis (at most one).
            Ex: [:, 1, ..., some_start : some_end : some_step]

        Output
        ----------
            - A subset of the array which matches the idx(s) requested.
        """
        selection = []

        ellipsis_counter = 0
        for i in range(len(idx)):
            if isinstance(idx[i], slice):
                new_selection: MySlice = self.normalize_slice(
                    idx[i].start, idx[i].stop, idx[i].step, i
                )
                selection.append(new_selection)
            elif isinstance(idx[i], int):
                if idx[i] >= 0:
                    selection.append(MySlice(idx[i], idx[i] + 1, 1))
                else:
                    selec0 = idx[i] + self.array.shape[i]
                    selection.append(MySlice(selec0, selec0 + 1, 1))
            elif isinstance(idx[i], type(Ellipsis)):
                if ellipsis_counter == 0:
                    new_selection: MySlice = self.normalize_slice(0, None, 1, i)
                    selection.append(new_selection)
                    ellipsis_counter += 1
                else:
                    # This is a possible bug in Dask:
                    # given a 3D dask array "a",
                    # a[:,:,:] works
                    # a[..., : , :] works
                    # a[..., ..., ...] gives an error
                    # a[..., ..., :] gives an error
                    # a[..., :, ...] gives an error
                    # so it seems to only handle one ellipse type. Therefore, to avoid upstream
                    # errors, we do the same.
                    raise ValueError("Only one use of Ellipsis allowed.")

        # build a dims specification type and set the selection to it.
        self.selection = DimsSpec(selection)
        return self.array.__getitem__(idx)

    def gc(self):
        """
        Garbage collect the DeisaArray by deleting it from memory.
        """
        del self.array


# TODO this class could probably be moved inside the Deisa since it just serves as a container of
# DeisaArray objects.
class DeisaArrays:
    """
    Container class of DeisaArray objects.
    """

    def __init__(self, arrays: dict[str, Array]):
        """
        Initialize DeisaArrays class.

        Parameters
        ----------
            - arrays: dictionary of key-value where the key is the name of the global array being
            shared and the value is the Dask Array.
            Ex:
            {
                "global_t": dask.Array(...)
                "temperature": dask.Array(...)
            }
        """
        self.arrays = []
        for shared_array_name, shared_array in arrays.items():
            self.arrays.append(DeisaArray(shared_array_name, shared_array))

        self.contract: Optional[ContractType] = None

    def __getitem__(self, name: str) -> DeisaArray:
        """
        Retrieve a specific DeisaArray with a name.

        Parameters
        ----------
            - name: name of DeisaArray we want to retrieve.

        Output
        ----------
            - DeisaArray corresponding to the desired name.
        """
        for deisa_array in self.arrays:
            if deisa_array.name == name:
                return deisa_array
        raise ValueError(f"{name} array does not exist in Deisa data store.")

    # TODO with the new solution, contracts are not handled yet.
    def handle_contract(self):
        """
        Generate and share the contract. Used in cases where the user will not change the contract
        during the course of the analytics.
        """
        self.generate_contract()
        self.share_contract()

    # TODO with the new solution, contracts are not handled yet.
    def generate_contract(self) -> ContractType:
        """
        Generate the contract. For each DeisaArray object, we store the name and selection as
        key-value pairs.

        Output
        ----------
            - A dictionary of key-value pairs consisting of the name of the array and the selection.
        """
        contract = {}
        for deisa_array in self.arrays:
            contract[deisa_array.name] = deisa_array.selection

        self.contract = ContractType(contract)
        print("Generated contract", self.contract, flush=True)
        return self.contract

    # TODO with the new solution, contracts are not handled yet.
    def share_contract(self):
        """
        Create a global variable that is shared among all clients effectively setting the contract.
        The contract is then read by PDI in the "is_contract_satisfied" method of the Bridge instance
        to know when/what to share.
        """
        Variable(CONTRACT_NAME).set(self.contract)
        print("Contract has been shared with all clients.", flush=True)

    def gc(self):
        """
        Garbage collect all the DeisaArray objects in the list.
        """
        for deisa_array in self.arrays:
            deisa_array.gc()


class Deisa:
    """
    The client-side connector to the simulation. Must be instantiated by the main analytics code.
    """

    def __init__(
        self,
        nb_workers: int,
        scheduler_file_name: str | None = None,
        scheduler_address: str | None = None,
        cluster=None,
        use_ucx: bool = False,
    ):
        """
        Initialize by loading a scheduler configuration, instantiating a
        client that connects to it, checking the versions, and obtaining the keys of the workers
        connected to the scheduler.

        Parameters
        ----------
            - nb_workers: number of workers the Adaptor expects will connect.

            - scheduler_file_name: the name of the scheduler config file in json format.

            - use_ucx: whether to use ucx.
        """
        if use_ucx:
            os.environ["DASK_DISTRIBUTED__COMM__UCX__INFINIBAND"] = "True"

        # TODO make nicer -- detect types
        if cluster:
            self.client = Client(cluster)
        elif scheduler_address:
            self.client = create_client_connected_to_scheduler_at(scheduler_address)
        elif scheduler_file_name:
            with open(scheduler_file_name, "r") as f:
                scheduler_config: dict = json.load(f)
            self.client: Client = create_client_connected_to_scheduler_at(
                scheduler_config["address"]
            )
        else:
            raise RuntimeError(
                "Must initialize Deisa with cluster object, scheudler encoding,"
                "or scheduler address."
            )

        # Check version info for the client, scheduler, and all the workers.
        # Raise error if there are any versions mismatch
        self.client.get_versions(check=True)

        # Get list of id of workers connected to scheduler.
        workers: list[Worker] = list(self.client.scheduler_info()[WORKERS_NAME].keys())

        # Ensure that all workers (expected) are connected to scheduler
        while len(workers) != nb_workers:
            workers = list(self.client.scheduler_info()[WORKERS_NAME].keys())

    def get_client(self) -> Client:
        """
        Return the client associated with the Adaptor.
        """
        return self.client

    def get_deisa_arrays(self) -> DeisaArrays:
        """
        Return DeisaArrays from the data that PDI shares.

        DeisaArrays is an object that handles a list of DeisaArray objects which are essentially
        dask arrays with additional features to track which data is needed and which is not.

        Output
        ----------
            - A DeisaArrays object.
        """

        # shared data will look something like this:
        # shared_data = {
        #     'global_t': {
        #         'timedim': 0,
        #         'subsizes': [1, 10, 20],
        #         'starts': [0, 0, 0],
        #         'sizes': [10, 20, 20]
        #         'dtype': "double"
        #     }
        #     'global_p': {
        #         'timedim': 0,
        #         'subsizes': [1, 20, 30],
        #         'starts': [0, 0, 0],
        #         'sizes': [30, 100, 6000]
        #         'dtype': "double"
        #     }
        # }
        shared_data: dict[str, dict] = Queue(ARRAYS_NAME, client=self.client).get()  # type: ignore
        assert isinstance(shared_data, dict)

        # create task_id dictionary from all the Queues being shared.
        # Each bridge shares a dict like:
        # {
        #     "name": ((Y,Z), "name-rankX") --- "(Y,Z)" and "X" varies per bridge
        #     "othername": ((Y,Z), "name-rankX") --- "(Y,Z)" and "X" varies per bridgeX"
        # }

        # We need to convert this to a single dictionry of this form:
        # {
        #     "name": {
        #         (Y1,Z1): "name-rankX",
        #         (Y2,Z2): "name-rankY",
        #         ...
        #     },
        #     "othername":{
        #         (Y1,Z1): "othername-rankX",
        #         (Y2,Z2): "othername-rankY",
        #         ...
        #     },
        #     ...
        # }

        size = Variable(NB_BRIDGES_NAME).get()
        self.task_id = defaultdict(dict)
        for i in range(size):  # type: ignore
            # for each MPI rank, get the dict of task + queue name per array being shared
            q: dict = Queue("task_id" + str(i)).get()  # type: ignore
            # for each array in dict, unpack and put it in the task_id dict
            for name, val in q.items():
                # val[0] is the task id -- (Y,Z)
                # val[1] is the queue name -- "name-rankX"
                self.task_id[name][val[0]] = val[1]

        arrays: dict[str, Array] = dict()
        for shared_array_name in shared_data.keys():
            # Manually create a dask array
            arrays[shared_array_name] = self.create_array(
                name=shared_array_name,
                shape=shared_data[shared_array_name][SIZE_NAME],
                chunksize=shared_data[shared_array_name][SUBSIZE_NAME],
                dtype=shared_data[shared_array_name][DTYPE_NAME],
                task_to_rank=self.task_id[shared_array_name],
            )

        return DeisaArrays(arrays)

    def create_array(self, name, shape, chunksize, dtype, task_to_rank):
        """
        Manually create a Dask Array from futures that represent the computations that will
        produced by each MPI process.
        The idea is that each MPI process will share part of the grid. Each of these computations
        are external futures which are chunks of the global array. We want to rebuild the global
        array from the collection of chunks so we can operate on it.

        Parameters
        ----------
            - name: The name of the array being shared by PDI.
            - shape: The global shape of the array including the dimension over which PDI is
            iterating.
            - chunksize: The desired chunksize. This corresponds to the dimension of data produced
            by each MPI process.
            - dtype: the data type of the array
            - task_to_rank: a dictionary where we associate a task ID to a queue name to get
            futures from

        Output
        ----------
            - A Dask array of the global data.
        """
        # TODO dask supports the creation of "sparse" arrays where only certain chunks are defined.
        # this is ok bc as long as all upstream tasks depend only on the chunks that are present.
        # A possible solution to the contract problem is to create *ONLY* the chunks requested by
        # the user!

        @dask.delayed
        def deisa_ext_task(pull_from, depends_on=None):
            # get a temporary client in the worker
            with worker_client():
                # get future of scatter operation from specific Queue
                f = Queue(pull_from).get()
            # return the result of the future
            # TODO return future directly? Investigate if Dask supports this in general.
            return f.result()  # type: ignore

        chunks_in_each_dim = [shape[i] // chunksize[i] for i in range(len(shape))]
        chunks = tuple(
            [(chunksize[i],) * chunks_in_each_dim[i] for i in range(len(shape))]
        )
        chunk_coords = list(itertools.product(*[range(i) for i in chunks_in_each_dim]))

        # chunk coords identify the task (except for the time dim)
        custom_gt = {}
        deps = []
        for coord in chunk_coords:
            # remove time dimension
            task_id = coord[1:]
            queue_name = task_to_rank[task_id]
            if coord[0] == 0:
                # if timestep is 0 (first time step) we simply create the external task.
                value = deisa_ext_task(pull_from=queue_name)
                # add dependency
                deps.append(value)
            else:
                # in all other cases, we create a fake time dependency by passing the previous task
                # as an argument. This makes sure that tasks get scheduled in the correct order.

                # TODO does it make a diff to depend on the ext task or the key of the task?
                # maybe this eliminates the problem of dask removing the time deps
                value = deisa_ext_task(
                    pull_from=queue_name,
                    depends_on=custom_gt[(name, coord[0] - 1, *coord[1:])],
                )
                deps.append(value)

            custom_gt[(name, *coord)] = value.key
        dsk = HighLevelGraph.from_collections(name, custom_gt, dependencies=deps)
        custom_gt = da.Array(dsk, name, chunks, dtype)
        return custom_gt

    def wait_for_last_bridge_and_shutdown(self):
        """
        Called by client to wait for all bridges to shutdown before
        trying to shutdown the main client.

        Parameters
        ----------
            - delay: how much time to wait before checking again if bridges are all shutdown.
        """
        assert self.client is not None
        self.client.shutdown()


class BridgeV1:
    """
    Bridge class for Deisa. It is a client that is initialized by each MPI process in the simulation.
    Each bridge connects to the scheduler and has a specific set of workers it will send data to.

    This is the V1 implementation which has a client associated per bridge. This creates overhead
    and limits the number of bridges we can have since Dask has a hardcoded limit.
    """

    def __init__(
        self,
        mpi_rank: int,
        mpi_size: int,
        arrays_description: dict[str, dict],
        arrays_description_dtype: dict,
        scheduler_encoding: list[int] | None = None,
        cluster=None,
        scheduler_address: str | None = None,
        use_ucx: bool = False,
    ):
        """
        Initialize a Bridge per MPI process.

        The Bridge receives a dictionary which describes the data that
        will be shared with Deisa client.

        Parameters
        ----------
            - scheduler_encoding: A list of character encodings for the file name of the scheduler
            configuration file..
            - mpi_rank: the mpi_rank of this Bridge
            - mpi_size: the total size of the MPI processes.
            - arrays_description: a dictionary where keys are the name of the arrays to be shared and the
            keys are dictionaries which describe each array that will be shared.
            For instance when sharing an array called "global_t", the dictionary might
            look like this:
                arrays_description = {
                    'global_t': {
                        'timedim': [0],
                        'subsizes': [1, 10, 20],
                        'starts': [0, 0, 0], # varying per MPI process!
                        'sizes': [10, 20, 20]
                    }
                    'global_p': {
                        'timedim': [0],
                        'subsizes': [1, 20, 100],
                        'starts': [0, 0, 0], # varying per MPI process!
                        'sizes': [30, 40, 100]
                    }
                }
            - arrays_description_dtype: a dictionary where keys are the name of the array and values are
            dtype of the underlying data being shared.
            For example:
                deisa_arrays_dtype = {
                    'global_t': double
                    'global_p': double
                    }
            - use_ucx: where to use UCX.

        """
        if scheduler_encoding is not None:
            scheduler_file_name: str = "".join(chr(i) for i in scheduler_encoding)
            with open(scheduler_file_name[:-1], "r") as f:
                scheduler_config: dict = json.load(f)
            address: str = scheduler_config["address"]
            self.client: Client = create_client_connected_to_scheduler_at(address)
        elif cluster is not None:
            self.client: Client = Client(cluster)
        elif scheduler_address is not None:
            self.client: Client = Client(scheduler_address)
        else:
            raise RuntimeError(
                "Must initialize Bridge with cluster object, scheudler encoding,"
                "or scheduler address."
            )

        self.mpi_rank: int = mpi_rank
        self.mpi_size: int = mpi_size

        # sanity check
        assert self.client is not None, "Client was not able to connect!"

        # get workers per bridge using round robin scheme
        self.workers: list[str] = self.get_workers()

        self.shared_data: dict[str, dict] = arrays_description
        self.shared_data_dtype: dict = arrays_description_dtype

        for array_name in self.shared_data.keys():

            # merge dtype info into description dict
            self.shared_data[array_name][DTYPE_NAME] = str(
                self.shared_data_dtype[array_name]
            )

            # unpack time dimension from [num] -> num
            self.shared_data[array_name][TIME_DIMENSION_NAME] = self.shared_data[
                array_name
            ][TIME_DIMENSION_NAME][0]

        if self.mpi_rank == 0:
            # share MPI size among all clients. I am sure that all of them are connected since
            # we have an assert above.
            self.nb_bridges = Variable(NB_BRIDGES_NAME, client=self.client).set(
                self.mpi_size
            )
            # Share the description. Since we only need info for size and subsize, only rank0
            # needs to share.
            Queue(ARRAYS_NAME).put(self.shared_data)

        # Contract of each bridge.
        self.contract = None

        # Each bridge has its own:
        # 1. Queue per array being shared - for each array, the queue will contain the
        # futures of scatter operation.
        # 2. Task_id per array i.e. task (0,0) is always associated to rank0 for example.
        # Hypothetically, this can be different for each array.
        self.queues = {}
        self.task_id = {}
        for name, v in self.shared_data.items():
            # position of bridge in global array: for example if bridge starts at position 6 in
            # dim1, and the subsize in dim1 is 2, then it will be the (6/2) 3rd bridge in that dim.
            new_k = [
                v["starts"][i] // v["subsizes"][i] for i in range(len(v["starts"]))
            ]
            # the id is time invariant. Each bridge will deal with same portion of array through
            # time. So we pop the time dimension. This is the task_id.
            new_k.pop(v[TIME_DIMENSION_NAME])

            # the name of the queue per bridge, per array being shared.
            qname = str(name) + "-rank" + str(self.mpi_rank)

            # for each array name, store the position of the array (invariant in time) and the name
            # of the queue.
            self.task_id[name] = (tuple(new_k), qname)

            # create the actual Queue with the name
            self.queues[name] = Queue(qname)

        # queues will be something like this (for bridge belonging to rankX):
        # {
        #   "global_t": Queue("global_t-rankX")
        #   "global_p": Queue("global_p-rankX")
        #   ....
        # }

        # task_id will be something like this:
        # {
        #   "global_t": ( (Y,Z), "global_t-rankX" )
        #   "global_p": ( (Y,Z), "global_p-rankX" )
        #   ....
        # }

        # Share task_id so that main client can build arrays properly.
        Queue("task_id" + str(self.mpi_rank)).put(self.task_id)

    def get_workers(self) -> list[str]:
        """
        Get the Dask workers that will receive data from the Bridge. The worker(s) are chosen based
        on the rank of the Bridge.

        Output
        ----------
            - List of workers corresponding to the Bridge.
        """
        total_dask_workers = list(self.client.scheduler_info()[WORKERS_NAME].keys())

        if self.mpi_size >= len(total_dask_workers):
            # more MPI processes than dask_workers - each MPI process sends only to one dask_worker
            # ex: MPI size 10 and total_dask_workers  5
            # rank 0 and rank 5 will send to worker 0
            # rank 1 and rank 6 will send to worker 1
            # etc.
            # This is a round robin scheme
            return [total_dask_workers[self.mpi_rank % len(total_dask_workers)]]
        else:
            raise RuntimeError(
                "There are more Dask workers than MPI processes. There must be less"
                "(or the equal) Dask workers than MPI processes. "
            )

    # TODO For now, contracts are not being used since it would deadlock the whole system.
    # introduce them later.
    def publish_data(
        self, shared_array: NDArray, shared_array_name: str, timestep: int, debug=False
    ):
        """
        This method is called from PDI's deisa plugin. It is responsible for recalculating
        the position of each Bridge within the global arrays and for calling the scatter method.

        Parameters
        ----------
            - shared_array: the ndarray we are sharing.
            - shared_array_name: the name of the array we are sharing.
            - timestep: the current timestep.
            - debug: weather debug mode is activated.
        """

        # insert a dimension at timedim position.
        # so for example an array of shape (2,5) becomes of shape (1,2,5) if the timedim is 0
        # needed by dask since we build the entire array
        shared_array = np.expand_dims(
            shared_array, self.shared_data[shared_array_name][TIME_DIMENSION_NAME]
        )

        # scatter data to assigned dask worker
        f = self.client.scatter(shared_array, direct=True, workers=self.workers)

        # put the future in the corresponding queue
        self.queues[shared_array_name].put(f)

    def release(self):
        # TODO remnant of previous version.
        pass
