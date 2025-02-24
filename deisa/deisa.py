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

from dask.array import Array  # type: ignore
from dask.distributed import Client, Queue, Variable, Worker, Lock, Event
from dask.distributed import wait, get_worker
from dask.distributed import worker_client
from dask.highlevelgraph import HighLevelGraph
import dask
import dask.array as da
from collections import namedtuple, defaultdict
from typing import NewType, Optional, Set, List, Tuple, Union, Dict
import warnings
import logging
import itertools
import json
import os
import time

# Dask related
WORKERS_NAME = "workers"
ARRAYS_METADATA = "arrays-metadata"

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

# A ValidContract is a list of MySlice types for each dimension of the array.
ValidContract = NewType("ValidContract", List[MySlice])
# A NullContract is just None - It cant be a list of None because when slicing, this is interpreted
# as [0,end,1]
NullContract = NewType("NullContract", None)
# A contract is either a ValidContract or a NullContract
Contract = Union[ValidContract, NullContract]


def mapping_mpi_procs_to_dask_workers(mpi_size: int, dask_workers: list[str]):
    """
    Determine mapping of MPI ranks to corresponding Dask workers. This function can be overriden
    by the user to whatever mapping they desire.

    Output
    ----------
        - Dictionary of MPI rank number to list of IP addresses of Dask Workers.
    """

    if len(dask_workers) > mpi_size:
        raise RuntimeError(
            "There are more Dask workers than MPI processes. There must be less"
            "(or the equal) Dask workers than MPI processes. "
        )

    mapping = {}
    for rank in range(mpi_size):
        mapping[rank] = [dask_workers[rank % len(dask_workers)]]

    return mapping


def create_client_connected_to_scheduler_at(
    scheduler_address: str, max_retries=10
) -> Client:
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
        warnings.warn(
            f"Failed to create a client connected to scheduler at {scheduler_address}"
            f"because of: \n{e}\nRetrying connection...\n"
        )
        if max_retries == 0:
            raise RuntimeError(
                "Unable to connect to scheduler. Make sure the scheduler is running. Exiting..."
            )
        return create_client_connected_to_scheduler_at(
            scheduler_address, max_retries=(max_retries - 1)
        )


def qname_from(array_name: str, rank: int) -> str:
    """Return queue name from array name and rank in fixed scheme."""
    return str(array_name) + "-rank" + str(rank)


def get_bridge_instance(
    sched_file: List[int],
    mpi_rank: int,
    mpi_size: int,
    arrays_description: Dict[str, Dict],
    arrays_description_dtype: Dict[str, DTypeLike],
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
        arrays_metadata=arrays_description,
        arrays_metadata_dtype=arrays_description_dtype,
        **kwargs,
    )


def create_null_contract_for_all_arrays(arrays_metadata) -> Dict[str, Contract]:
    """
    Creates a null contract for all arrays. In the beginning this is the default for all
    arrays being shared.
    """
    arrays_contract: Dict[str, Contract] = {}
    for name in arrays_metadata:
        arrays_contract[name] = NullContract(None)
    return arrays_contract


def normalize_slice(
    slice_start: int | None,
    slice_end: int | None,
    slice_step: int | None,
    shape_at_dim: int,
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

    if slice_start is None:
        slice_start = 0
    elif slice_start < 0:
        slice_start = shape_at_dim + slice_start

    if slice_end is None:
        slice_end = shape_at_dim
    elif slice_end < 0:
        slice_end = shape_at_dim + slice_end

    if slice_step is None:
        slice_step = 1
    elif slice_step < 0:
        raise ValueError(f"{slice_step} only positive step values are accepted")

    return MySlice(slice_start, slice_end, slice_step)


def create_valid_contract(
    array_name: str, keys: tuple, array_metadata: Dict
) -> ValidContract:
    """
    Support basic dask syntax for slicing and sets the selection variable which is used
    to generate a contract.

    Parameters
    ----------
        - array_name: name of the array.
        - keys: a tuple of indexes the user requested (per array dimension).
          Ex: (:, 1, ..., some_start : some_end : some_step)
        - array_metadata: the metadata of the array.

    Output
    ----------
        - A subset of the array which matches the keys requested.
    """
    selection = []

    error_msg = f""" Only {len(keys)} dimensions selected for {array_name}. Array is 
        {len(array_metadata[SIZE_NAME])}-dimensional. Please select data in all dimensions 
        (':' is accepted).
    """

    assert len(keys) == len(array_metadata[SIZE_NAME]), error_msg

    ellipsis_counter = 0
    for i in range(len(keys)):
        if isinstance(keys[i], slice):
            new_selection: MySlice = normalize_slice(
                keys[i].start,
                keys[i].stop,
                keys[i].step,
                int(array_metadata[SIZE_NAME][i]),
            )
            selection.append(new_selection)
        elif isinstance(keys[i], int):
            if keys[i] >= 0:
                selection.append(MySlice(keys[i], keys[i] + 1, 1))
            else:
                selec0 = keys[i] + int(array_metadata[SIZE_NAME][i])
                selection.append(MySlice(selec0, selec0 + 1, 1))
        elif isinstance(keys[i], type(Ellipsis)):
            if ellipsis_counter == 0:
                selection.append(MySlice(0, int(array_metadata[SIZE_NAME][i]), 1))
                ellipsis_counter += 1
            else:
                # Dask only allows 1 ellipsis, so we maintain a similar API.
                raise ValueError("Only one use of Ellipsis allowed.")
        else:
            raise RuntimeError(
                "Only slice, int, or at ellipsis (at most one) is allowed in data selection."
            )

    return ValidContract(selection)


def efficient_chunks_for_dimension(
    start: int, end: int, step: int, chunk_size: int
) -> Set[int]:
    """
    Compute the set of chunk indices touched by a slice in one dimension
    without iterating over every slice element. This version uses a for loop
    over candidate chunk indices.

    The slice is defined by:
      - start: start index
      - end: stop index (exclusive) [assumed b > a]
      - step: step (assumed positive)

    chunk_size is the chunk size along this dimension. An array index i belongs to chunk i // d.

    Returns:
      A set of chunk indices that the arithmetic progression
      (start, start+step, start+2step, …, end) touches.
    """
    # Compute the total number of elements in the slice using ceiling division.
    # This is equivalent to math.ceil((b - a) / c)
    n: int = (end - start + step - 1) // step

    # Determine the first chunk index.
    first_chunk: int = start // chunk_size

    # Compute the last index in the slice and its chunk.
    last_index: int = start + (n - 1) * step
    last_chunk: int = last_index // chunk_size

    chunk_indices: Set[int] = set()

    # Iterate over candidate chunk indices from first_chunk to last_chunk.
    for j in range(first_chunk, last_chunk + 1):
        if j == first_chunk:
            # The first chunk is always touched since a is in it.
            chunk_indices.add(j)
        else:
            # For chunk j, we need the first slice element that reaches or exceeds j*d.
            # We compute the smallest k (position in the slice) such that:
            #    a + k*c >= j*d
            # Using ceiling division with integer arithmetic:
            k: int = ((j * chunk_size - start) + step - 1) // step
            # Ensure that k is within the slice and that the element belongs to chunk j.
            if k < n and (start + k * step) // chunk_size == j:
                chunk_indices.add(j)

    return chunk_indices


def needed_chunks(
    contract: ValidContract,
    chunk_shape: Tuple[int, ...],
) -> List[Tuple[int, ...]]:
    """
    Determine the multi-dimensional chunk coordinates needed to cover all indices
    specified by a tuple of slices. The array has shape 'array_shape' and is partitioned
    into chunks of shape 'chunk_shape'. Each slice in 'slices' selects indices along
    its respective dimension.

    Parameters:
      slices: A tuple of slice objects (one per dimension).
      array_shape: The overall shape of the array.
      chunk_shape: The shape (i.e. chunk sizes) along each dimension.

    Returns:
      A list of tuples where each tuple represents the coordinate of a chunk that
      contains at least one element from the specified slices.
    """
    chunks_per_dim: List[Set[int]] = []
    # Process each dimension individually.
    for dim, s in enumerate(contract):
        chunk_set: Set[int] = efficient_chunks_for_dimension(
            s.start, s.stop, s.step, chunk_shape[dim]
        )
        # Sorting for consistent ordering.
        chunks_per_dim.append(chunk_set)

    # The overall needed chunks are the Cartesian product of chunk indices from each dimension.
    return list(itertools.product(*chunks_per_dim))


class Deisa:
    """
    The client-side connector to the simulation. Must be instantiated by the main analytics code.
    """

    def __init__(
        self,
        nb_expected_dask_workers: int,
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

            - scheduler_file_name: the name of the scheduler config file in json format. Useful
            when the scheduler is started from the CLI with the --scheduler-file flag.

            - scheduler_address: the address of the scheduler. Useful when the scheduler is started
            from the CLI with the --scheduler-address flag.

            - cluster: a Dask cluster instance. Useful when you start a cluster from the python
            script.

            - use_ucx: whether to use ucx.
        """
        if use_ucx:
            os.environ["DASK_DISTRIBUTED__COMM__UCX__INFINIBAND"] = "True"

        if cluster:
            self._client = Client(cluster)
        elif scheduler_address:
            self._client = create_client_connected_to_scheduler_at(scheduler_address)
        elif scheduler_file_name:
            with open(scheduler_file_name, "r") as f:
                scheduler_config: Dict = json.load(f)
            self._client: Client = create_client_connected_to_scheduler_at(
                scheduler_config["address"]
            )
        else:
            raise RuntimeError(
                "Must initialize Deisa with cluster object, scheduler file,"
                "or scheduler address."
            )

        # Get list of id of workers connected to scheduler.
        self.connected_dask_workers: List[str] = list(
            self._client.scheduler_info()[WORKERS_NAME].keys()
        )

        # Ensure that all workers (expected) are connected to scheduler
        while len(self.connected_dask_workers) != nb_expected_dask_workers:
            self.connected_dask_workers = list(
                self._client.scheduler_info()[WORKERS_NAME].keys()
            )

        # blocking call from rank0
        self.mpi_size: int = Variable(NB_BRIDGES_NAME).get()  # type: ignore

        self.mapping_rank_to_workers = mapping_mpi_procs_to_dask_workers(
            self.mpi_size, self.connected_dask_workers
        )

        # get metadata of arrays being shared (blocking call from rank0)
        self.arrays_metadata: Dict[str, Dict] = Queue(ARRAYS_METADATA, client=self._client).get()  # type: ignore
        # arrays_metadata will look something like this:
        # arrays_metadata = {
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

        # create contract -- By default, at initialization, the contract is invalid
        self.arrays_contract: Dict[str, Contract] = create_null_contract_for_all_arrays(
            self.arrays_metadata
        )

        # initialization of mapping of task ID to Queue name for each array
        self.map_taskID_qname: Dict[str, Dict[tuple, tuple]] = defaultdict(dict)

        # update the mapping
        self.create_map_taskID_qname()

        # block simulation
        self.block()

    @property
    def client(self) -> Client:
        """
        Return the client associated with the Adaptor.
        """
        return self._client

    def upate_metadata(
        self,
    ):
        """Function that can be called from PDI to update the metadata."""
        logging.info("Updating metadata!")
        pass

    def report_event(self, event):
        """
        Function to report the ocurrence of an interesting event.
        """
        logging.info(f"Event {event} has ocurred!")
        pass

    def create_map_taskID_qname(self):
        """
        Each MPI rank shares a Queue that contains a dictionary that maps array names to a tuple
        of (TaskID, RankNum). For two ranks:
            rankX = {
                "arrayname1" : (taskID-Arr1-X, RankNumX)
                "arrayname2" : (taskID-Arr2-X, RankNumX)
            }
            rankY = {
                "arrayname1" : (taskID-Arr1-Y, RankNumY)
                "arrayname2" : (taskID-Arr2-Y, RankNumY)
            }
        We want to create a single dictionary that all the information for each array and gets the
        qname for each array based on the rank. I.e:
        {
            "arrayname1": {
                taskID-Arr1-X: qname-rankNumX
                taskID-Arr1-Y: qname-rankNumY
                ...
            },
            "arrayname2": {
                taskID-Arr2-X: qname-rankNumX
                taskID-Arr2-Y: qname-rankNumY
                ...
            }
        }

        This corresponds to a dictionary inversion of some sort.
        """

        # create task_id dictionary from all the Queues being shared.
        # Each bridge shares a dict like:
        # {
        #     "name":      ((Y,Z), X) --- "(Y,Z)" and "X" varies per bridge
        #     "othername": ((Y,Z), X) --- "(Y,Z)" and "X" varies per bridgeX"
        # }

        # We need to convert this to a single dictionry of this form (taskid : Queue_name):
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

        for i in range(self.mpi_size):  # type: ignore
            # for each MPI rank, get the dict of task + rank name per array being shared
            # blocking call
            d: Dict = Queue("task_id" + str(i)).get()  # type: ignore
            # for each array in dict, unpack and put it in the task_id dict
            for name, val in d.items():
                # val[0] is the task id -- (Y,Z)
                # val[1] is the rank number -- X
                self.map_taskID_qname[name][val[0]] = (qname_from(name, val[1]), val[1])

    def _create_array(
        self,
        name: str,
        shape: List[int],
        chunkshape: List[int],
        dtype: str,
        task_to_rank: Dict,
        rank_to_workers: Dict,
        contract: ValidContract,
    ):
        """
        Manually create a Dask Array from futures that represent the computations that will
        produced by each MPI process.
        Each MPI process will share part of the grid. Each of these computations
        are external futures which are chunks of the global array. We want to rebuild the global
        array from the collection of chunks so we can operate on it.

        Parameters
        ----------
            - name: The name of the array.
            - shape: The shape of the entire array including the time dimension.
            - chunksize: The shape of each chunk. This corresponds to the shape of a subgrid
            produced by an MPI process.
            - dtype: the data type of the array.
            - task_to_rank: a mapping of taskID to a queue name to get futures from.
            - rank_to_workers: a mapping of MPI rank to Dask Workers.
            - contract: a ValidContract that specifies what index per dimension the user wants for
            the array.

        Output
        ----------
            - A Dask array of the global data.
        """

        @dask.delayed  # type: ignore
        def deisa_ext_task(task_id, pull_from, depends_on=None):
            # get a temporary client in the worker
            with worker_client():
                # get future of scatter operation from specific Queue
                f = Queue(pull_from).get()
                # print(f"Executing {task_id} on worker {get_worker().name}")
            return f.result()  # type: ignore

        chunks_in_each_dim = [shape[i] // chunkshape[i] for i in range(len(shape))]
        chunks = tuple(
            [(chunkshape[i],) * chunks_in_each_dim[i] for i in range(len(shape))]
        )
        needed_chunk_coords = needed_chunks(
            contract=contract, chunk_shape=tuple(chunkshape)
        )

        # TODO make sure its sorted. For now we assume it is because of how
        # itertools.product works. But in the future, it would be nice to make sure.

        # chunk coords identify the task (except for the time dim)
        custom_gt = {}
        deps = []
        first_needed_time = needed_chunk_coords[0][0]
        last_task_id = {}
        for coord in needed_chunk_coords:
            # remove time dimension
            task_id = coord[1:]
            queue_name = task_to_rank[task_id][0]
            dask_worker = rank_to_workers[task_to_rank[task_id][1]]
            if coord[0] == first_needed_time:
                # for first needed time, just create the ext task.
                with dask.annotate(workers=dask_worker, allow_other_workers=False):  # type: ignore
                    value = deisa_ext_task(task_id=task_id, pull_from=queue_name)

                # add the dependency
                deps.append(value)
                last_task_id[task_id] = value
            else:
                # in all other cases, we create a fake time dependency by passing the previous tasks
                # as an argument. This makes sure that tasks get scheduled in the correct order.
                with dask.annotate(workers=dask_worker, allow_other_workers=False):  # type: ignore
                    value = deisa_ext_task(
                        task_id=task_id,
                        pull_from=queue_name,
                        depends_on=last_task_id[task_id],
                    )
                deps.append(value)
                last_task_id[task_id] = value

            custom_gt[(name, *coord)] = value.key
        dsk = HighLevelGraph.from_collections(name, custom_gt, dependencies=deps)
        custom_gt = da.Array(dsk, name, chunks, dtype)  # type: ignore
        return custom_gt

    def __getitem__(self, keys: tuple) -> Array:
        """
        Entry point for Deisa adaptor. Builds the array with a given name and a given slice.
        """
        # extract name of array
        assert isinstance(
            keys[0], str
        ), f"Expected str (for array name) as first argument of __getitem__(). Got {type(keys[0])}"

        name = keys[0]

        try:
            # try to fetch metadata for array
            metadata = self.arrays_metadata[name]
        except KeyError:
            # if array name does not exist in metadata, raise KeyError - This is the control aspect
            # of the contract paper.
            raise KeyError(
                f"Array {name} is not shared by the simulation. Check that"
                "simulation.yml is properly configured to share the array."
            )
        else:
            assert len(keys[1:]) == len(
                self.arrays_metadata[name][SIZE_NAME]
            ), f"""
            Expected a slice in all dimensions of array {name}. Received {len(keys[1:])}. Please 
            specify slice in all dimensions, for example: [:,:,:] for a 3D array.
            """
            logging.info("Creating contract...")
            contract: ValidContract = create_valid_contract(
                name, tuple(keys[1:]), metadata
            )

            if type(self.arrays_contract[name]) is ValidContract:
                logging.info("User requested new contract... Updating.")
            else:
                logging.info("Creating new contract for user!")

            # update the contract for the array internally
            self.arrays_contract[name] = contract

            # At this point, the metadata is set, the contract is set.
            # We can create the (sparse) array!
            return self._create_array(
                name=name,
                # I need to know shape of the array
                shape=metadata[SIZE_NAME],
                # I need to know the shape of subgrids shared by each MPI rank
                chunkshape=metadata[SUBSIZE_NAME],
                # I need to know the dtype of the array
                dtype=metadata[DTYPE_NAME],
                # I need the mapping of task_id to Queue name
                task_to_rank=self.map_taskID_qname[name],
                # I need the mapping of rank_to_workers
                rank_to_workers=self.mapping_rank_to_workers,
                # I need the contract for this specific array
                contract=contract,
            ).__getitem__(keys[1:])

    def share_contract(self):
        """
        Share the contract with Bridges. Called everytime __getitem__ is called so that
        Bridges update the contract.
        """
        Variable("contract").set(self.arrays_contract)
        logging.info("contract shared with bridges.")

    def block(
        self,
    ):
        Variable("block").set(True)

    def ready(
        self,
    ):
        self.share_contract()
        Variable("block").set(False)

    def wait_for_last_bridge_and_shutdown(self, delay=2):
        """
        Called by client to wait for all bridges to shutdown before
        trying to shutdown the main client.

        Parameters
        ----------
            - delay: how much time to wait before checking again if bridges are all shutdown.
        """
        assert self._client is not None
        nb_bridges_still_active = Variable(NB_BRIDGES_NAME).get()
        if nb_bridges_still_active == 0:
            # shutdown last client and whole cluster
            self._client.shutdown()
        else:
            time.sleep(delay)
            self.wait_for_last_bridge_and_shutdown(delay=delay)

        # TODO data cleanup


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
        arrays_metadata: Dict[str, Dict],
        arrays_metadata_dtype: Dict,
        scheduler_encoding: List[int] | None = None,
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
                scheduler_config: Dict = json.load(f)
            address: str = scheduler_config["address"]
            self.client: Client = create_client_connected_to_scheduler_at(address)
        elif cluster is not None:
            self.client: Client = Client(cluster)
        elif scheduler_address is not None:
            self.client: Client = Client(scheduler_address)
        else:
            raise RuntimeError(
                "Must initialize Bridge with cluster object, scheduler encoding,"
                "or scheduler address."
            )

        self.mpi_rank: int = mpi_rank
        self.mpi_size: int = mpi_size

        # sanity check
        assert self.client is not None, "Client was not able to connect!"

        # get workers per bridge using round robin scheme
        mapping_mpi_to_dask: Dict[int, list] = mapping_mpi_procs_to_dask_workers(
            self.mpi_size, list(self.client.scheduler_info()[WORKERS_NAME].keys())
        )
        self.dask_workers: List[str] = mapping_mpi_to_dask[self.mpi_rank]

        self.arrays_metadata: Dict[str, Dict] = arrays_metadata

        for array_name in self.arrays_metadata.keys():

            # merge dtype info into description dict
            self.arrays_metadata[array_name][DTYPE_NAME] = str(
                arrays_metadata_dtype[array_name]
            )

            # unpack time dimension from [num] -> num
            self.arrays_metadata[array_name][TIME_DIMENSION_NAME] = (
                self.arrays_metadata[array_name][TIME_DIMENSION_NAME][0]
            )

        if self.mpi_rank == 0:
            # share MPI size among all clients. I am sure that all of them are connected since
            # we have an assert above.
            self.nb_bridges = Variable(NB_BRIDGES_NAME, client=self.client).set(
                self.mpi_size
            )
            # Share the description. Since we only need info for size and subsize, only rank0
            # needs to share.
            Queue(ARRAYS_METADATA).put(self.arrays_metadata)

        # Each bridge has its own:
        # 1. Queue per array being shared - for each array, the queue will contain the
        # futures of scatter operation.
        # 2. Task_id per array i.e. task (0,0) is always associated to rank0 for example.
        # Hypothetically, this can be different for each array.
        self.queues: Dict[str, Queue] = {}
        task_id: Dict[str, tuple[tuple, int]] = {}
        self.current_chunk_per_array: Dict[str, list] = {}
        for name, v in self.arrays_metadata.items():
            # position of bridge in global array: for example if bridge starts at position 6 in
            # dim1, and the subsize in dim1 is 2, then it will be the (6/2) 3rd bridge in that dim.
            new_k = [
                v["starts"][i] // v["subsizes"][i] for i in range(len(v["starts"]))
            ]
            # add chunk coord info for that array
            self.current_chunk_per_array[name] = list(new_k)

            # the id is time invariant. Each bridge will deal with same portion of array through
            # time. So we pop the time dimension. This is the task_id.
            new_k.pop(v[TIME_DIMENSION_NAME])

            # for each array name, store the position of the array (invariant in time) and the name
            # of the queue.
            task_id[name] = (tuple(new_k), self.mpi_rank)

            # the name of the queue per bridge, per array being shared.
            # The name must be unique across ranks otherwise all ranks will put in the same queue.
            # (the queue is globally shared and is identified by its name)
            qname = qname_from(name, self.mpi_rank)

            # create the actual Queue with name qname
            self.queues[name] = Queue(qname)

        # queues will be something like this (for bridge belonging to rankX):
        # {
        #   "global_t": Queue("global_t-rankX")
        #   "global_p": Queue("global_p-rankX")
        #   ....
        # }

        # task_id will be something like this (for bridge belonging to rankX):
        # {
        #   "global_t": ( (Y,Z), X )
        #   "global_p": ( (Y,Z), X )
        #   ....
        # }

        # Share task_id so that main client can build arrays properly.
        Queue("task_id" + str(self.mpi_rank)).put(task_id)

        # wait until analytics are ready.
        while Variable("block").get():
            time.sleep(1)

        self.arrays_contract: Dict[str, Contract] = {}
        d_c: Dict[str, ValidContract] = Variable("contract").get()  # type: ignore
        for name, contract in d_c.items():
            if contract is None:
                self.arrays_contract[name] = NullContract(None)
            else:
                self.arrays_contract[name] = ValidContract(
                    list(map(MySlice._make, contract))
                )

        # based on contract, we know what are the needed chunks for each array
        self.simulation_needed_chunks = {}
        for name, contract in self.arrays_contract.items():
            if type(contract) is NullContract:
                self.simulation_needed_chunks[name] = None
            else:
                self.simulation_needed_chunks[name] = needed_chunks(
                    contract=contract,
                    chunk_shape=self.arrays_metadata[name][SUBSIZE_NAME],
                )

    def publish_data(self, array: NDArray, array_name: str, timestep: int, debug=False):
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
        # TODO
        # publish_data has missing behavior:
        # 1. when analytics calls block to simulation it should wait (user might be requesting new
        # analytics)
        # 2. when interesting event happens, simulation will roll back to a previous state and new
        # analytics will be triggered. In this case publish data should just return for the current
        # timestep.
        # WE MIGHT NEED TO CLEAN UP ANY LINGERING FUTURES
        # while Variable("block").get():
        #     time.sleep(1)
        # if important_event():
        #     return False

        # update the current chunk
        self.current_chunk_per_array[array_name][0] = timestep

        # check if data is needed
        data_is_needed: bool = self.check_data_is_needed(array_name, timestep)

        if data_is_needed:

            # insert a dimension at timedim position.
            # so for example an array of shape (2,5) becomes of shape (1,2,5) if the timedim is 0
            # needed by dask since we build the entire array
            array = np.expand_dims(
                array,
                self.arrays_metadata[array_name][TIME_DIMENSION_NAME],
            )

            # scatter data to assigned dask worker
            f = self.client.scatter(array, direct=True, workers=self.dask_workers)

            # put the future in the corresponding queue
            self.queues[array_name].put(f)
        else:
            logging.info("Data not needed by analytics.")
            pass

    def check_data_is_needed(
        self,
        array_name,
        timestep,
    ):

        if self.simulation_needed_chunks[array_name] is None:
            return False
        elif (
            tuple(self.current_chunk_per_array[array_name])
            in self.simulation_needed_chunks[array_name]
        ):
            logging.info(
                f"Sharing {array_name} at t={timestep} from rank {self.mpi_rank}."
            )
            return True
        else:
            logging.info(
                f"""
                Trying to share {array_name} at t={timestep} from rank {self.mpi_rank} but user 
                does not need it. Share avoided."""
            )
            return False

    def release(self):
        with Lock(BRIDGE_LOCK_NAME, client=self.client):
            # shut down client gracefully
            # get number of bridges to reduce by one
            var_nb_bridges = Variable(NB_BRIDGES_NAME)
            nb_bridges = var_nb_bridges.get(timeout="500ms")

            # reduce number of bridges by one since this client shut down
            var_nb_bridges.set(nb_bridges - 1)  # type: ignore

        # this call has to be outside of with context manager otherwise I get IO loop closed error.
        # basically the client has to inform the scheduler that lock is free, but if I close the
        # client this is impossible to do.
        self.client.close(timeout=5)
