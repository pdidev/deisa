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

import itertools
import json
import os
import time
import trace

import dask
import dask.array as da
import numpy as np
from dask.array import Array
from dask.distributed import Client, Queue, Future, Variable, Lock

from deisa.__version__ import __version__

DASK_VARIABLE_NAME_WORKERS = "workers"
DASK_LOCK_NB_BRIDGES = "nb-bridges-lock"
DASK_VARIABLE_NB_BRIDGES = "nb-bridges"
DASK_QUEUE_NAME_QUEUE = "queue"

DEISA_ARRAY_SIZE = "sizes"
DEISA_ARRAY_SUBSIZE = "subsizes"
DEISA_ARRAY_DTYPE = "dtype"
DEISA_ARRAY_TIME_DIMENSION = "timedim"
DEISA_ARRAY_START = "starts"


class Adaptor:
    """
    Deisa Adaptor Class.
    This class is instantiated in the user's analytics code.
    """

    adr = ""
    client = None
    workers = []

    def __init__(self, nb_workers: int, scheduler_info):
        with open(scheduler_info) as f:
            s = json.load(f)
        self.adr = s["address"]
        try:
            self.client = Client(self.adr)
        except Exception as _:
            # TODO: retry N times.
            print("retrying ...", flush=True)
            self.client = Client(self.adr)
 
        # Check if client version is compatible with scheduler version
        self.client.get_versions(check=True)
        # dask.config.set({
        #   "distributed.deploy.lost-worker-timeout": 60,
        #   "distributed.workers.memory.spill":0.97,
        #   "distributed.workers.memory.target":0.95,
        #   "distributed.workers.memory.terminate":0.99
        # })
        self.workers = list(self.client.scheduler_info()["workers"].keys())
        while (len(self.workers) != nb_workers):
            self.workers = list(self.client.scheduler_info()["workers"].keys())
        Variable(DASK_VARIABLE_NAME_WORKERS).set(self.workers)
        print(self.workers, flush=True)

    def get_client(self):
        return self.client

    def create_array(self, name, shape, chunksize, dtype, timedim):
        chunks_in_each_dim = [shape[i] // chunksize[i]
                              for i in range(len(shape))]
        lst = list(itertools.product(*[range(i) for i in chunks_in_each_dim]))
        items = []
        for m in lst:
            f = Future(key=("external-" + name, m), inform=True, external=True)
            d = da.from_delayed(dask.delayed(f), shape=chunksize, dtype=dtype)
            items.append([list(m), d])
        ll = self.array_sort(items)
        arrays = da.block(ll)
        return arrays

    # list arrays, one for each time step.
    def create_array_list(self, name, shape, chunksize, dtype, timedim):
        chunks_in_each_dim = [shape[i] // chunksize[i]
                              for i in range(len(shape))]
        lst = list(itertools.product(*[range(i) for i in chunks_in_each_dim]))
        items = []
        for m in lst:
            f = Future(key=("external-" + name, m), inform=True, external=True)
            d = da.from_delayed(dask.delayed(f), shape=chunksize, dtype=dtype)
            items.append([list(m), d])
        ll = self.array_sort(items)
        for i in ll:
            arrays.append(da.block(i))
        return arrays

    def array_sort(self, ListDs):
        if len(ListDs[0][0]) == 0:
            return ListDs[0][1]
        else:
            dico = dict()
            for e in ListDs:
                dico.setdefault(e[0][0], []).append([e[0][1:], e[1]])
            return [self.array_sort(dico[k]) for k in sorted(dico.keys())]

    def get_dask_arrays(self, as_list=False):  # TODO test
        arrays = dict()
        arrays_desc = Queue("Arrays").get()
        for name in arrays_desc:
            if not as_list:
                arrays[name] = self.create_array(
                    name,
                    arrays_desc[name][DEISA_ARRAY_SIZE],
                    arrays_desc[name][DEISA_ARRAY_SUBSIZE],
                    arrays_desc[name][DEISA_ARRAY_DTYPE],
                    arrays_desc[name][DEISA_ARRAY_TIME_DIMENSION]
                )
            else:
                arrays[name] = self.create_array_list(
                    name,
                    arrays_desc[name][DEISA_ARRAY_SIZE],
                    arrays_desc[name][DEISA_ARRAY_SUBSIZE],
                    arrays_desc[name][DEISA_ARRAY_DTYPE],
                    arrays_desc[name][DEISA_ARRAY_TIME_DIMENSION]
                )
        return arrays

    def get_deisa_arrays(self):
        assert(self.client is not None)
        arrays = dict()
        arrays_desc = Queue("Arrays", client=self.client).get()
        for name in arrays_desc:
            arrays[name] = self.create_array(
                name,
                arrays_desc[name][DEISA_ARRAY_SIZE],
                arrays_desc[name][DEISA_ARRAY_SUBSIZE],
                arrays_desc[name][DEISA_ARRAY_DTYPE],
                arrays_desc[name][DEISA_ARRAY_TIME_DIMENSION]
            )
        return deisa_arrays(arrays)
    
    def wait_for_last_bridge_and_shutdown(self, delay=2):
        assert(self.client is not None)

        nb_bridge = Variable(DASK_VARIABLE_NB_BRIDGES).get()
        if nb_bridge == 0:
            self.client.shutdown()
        else:
            time.sleep(delay)
            self.wait_for_last_bridge_and_shutdown(delay=delay)


class Deisa(Adaptor):
    """
    Instantiated in the client's analytics code.
    """

    def __init__(self, scheduler_info, nb_workers: int, use_ucx=False):
        if use_ucx:
            os.environ["DASK_DISTRIBUTED__COMM__UCX__INFINIBAND"] = "True"

        super().__init__(nb_workers, scheduler_info)


def get_bridge_instance(sched_file, rank, size, arrays, deisa_arrays_dtype, **kwargs):
    """
    Helper function to instantiate Bridge class from C++ (pybind11)
    :param kwargs: parameters that are unpacked in Bridge constructor
    :return: Bridge instance
    """
    return Bridge(sched_file, rank, size, arrays, deisa_arrays_dtype, **kwargs)


class Bridge:
    """
    Deisa Bridge class
    """

    def __init__(self, sched_file, rank, size, arrays, deisa_arrays_dtype, use_ucx=False):
        if use_ucx:
            os.environ["DASK_DISTRIBUTED__COMM__UCX__INFINIBAND"] = "True"

        self.client = self.__connect(sched_file)
        self.rank = rank
        self.contract = None

        assert(self.client is not None)
        self.nb_bridges = Variable(DASK_VARIABLE_NB_BRIDGES, client=self.client)

        workers = Variable(DASK_VARIABLE_NAME_WORKERS).get()
        if size > len(workers):  # more processes than workers
            self.workers = [workers[rank % len(workers)]]
        else:
            k = len(workers) // size  # more workers than processes
            self.workers = workers[rank * k:rank * k + k]

        self.arrays = arrays
        for ele in self.arrays:
            self.arrays[ele][DEISA_ARRAY_DTYPE] = str(deisa_arrays_dtype[ele])
            self.arrays[ele][DEISA_ARRAY_TIME_DIMENSION] = self.arrays[ele][DEISA_ARRAY_TIME_DIMENSION][0]
            self.position = [self.arrays[ele][DEISA_ARRAY_START][i] // self.arrays[ele][DEISA_ARRAY_SUBSIZE][i]
                             for i in range(len(np.array(self.arrays[ele][DEISA_ARRAY_SIZE])))]
        if rank == 0:
            # If and only if I have a perfect domain decomposition
            Queue("Arrays").put(self.arrays)
            self.nb_bridges.set(1)
        else:
            with Lock(DASK_LOCK_NB_BRIDGES, client=self.client):
                i = self.nb_bridges.get()
                self.nb_bridges.set(i+1)

    def release(self):
        if self.client and self.nb_bridges:
            print("release called", flush=True)
            with Lock(DASK_LOCK_NB_BRIDGES, client=self.client):
                self.nb_bridges.set(self.nb_bridges.get(timeout="500ms") - 1) # note: this should never throw a timeouterror because nb_bridges is always set
                print("nb bridges=" + str(self.nb_bridges.get()))

    def create_key(self, name):
        position = tuple(self.position)
        return "external-" + name, position

    def publish_request(self, data_name, timestep):
        try:
            selection = self.contract[data_name]
        except KeyError:
            return False

        self.position[self.arrays[data_name][DEISA_ARRAY_TIME_DIMENSION]] = timestep

        if selection == "All":
            return True
        elif selection is None:
            return False
        elif isinstance(selection, (list, tuple)):
            starts = np.array(self.arrays[data_name][DEISA_ARRAY_START])
            ends = np.array(self.arrays[data_name][DEISA_ARRAY_START]) + np.array(self.arrays[data_name][DEISA_ARRAY_SUBSIZE])
            sizes = np.array(self.arrays[data_name][DEISA_ARRAY_SUBSIZE])

            # if not needed timestep
            if (timestep >= selection[0][1]
                    or timestep < selection[0][0]
                    or (timestep - selection[0][0]) % selection[0][2] != 0):
                return False
            else:  # wanted timestep
                for i in range(1, len(selection)):
                    s = selection[i]  # i is dim
                    if starts[i] >= s[1] or ends[i] < s[0] or (ends[i] % s[2]) > sizes[i]:
                        return False
                # if there is at least some data for a dim
            return True

    """
    Called from PDI's deisa plugin.
    This method should not print anything to stdout.
    """
    def publish_data(self, data, data_name, timestep, debug=__debug__):

        if self.contract is None:
            self.contract = Variable("Contract").get()

        publish = self.publish_request(data_name, timestep)
        if publish:
            key = self.create_key(data_name)
            shap = list(data.shape)
            new_shape = tuple(
                shap[:self.arrays[data_name][DEISA_ARRAY_TIME_DIMENSION]] + [1] + shap[self.arrays[data_name][DEISA_ARRAY_TIME_DIMENSION]:])
            # TODO will not copy, if not possible raise an error so handle it :p
            data.shape = new_shape

            if debug:
                ts = time.time()
                tracer = trace.Trace(count=0, trace=0, countfuncs=1, countcallers=1)

                self.__scatter(data, key)

                allstats = "stats_r" + str(self.rank) + ".t" + str(timestep)
                debug = "debug_r" + str(self.rank) + ".t" + str(timestep)
                callgrind = "callgrind_r" + str(self.rank) + ".t" + str(timestep)

                ts = time.time() - ts
                print("scatter et profiling: ", ts, "s", flush=True)
            else:
                self.__scatter(data, key)

            data = None
        else:
            # print(data_name, "is not shared from process", self.position, " in timestep", timestep, flush=True)
            pass

    @staticmethod
    def __connect(sched_file):
        sched = ''.join(chr(i) for i in sched_file)
        with open(sched[:-1]) as f:
            s = json.load(f)
        adr = s["address"]
        try:
            client = Client(adr)
        except Exception as _:
            print("retrying ...", flush=True)  # TODO: retry N times
            client = Client(adr)
        return client

    def __scatter(self, data, key):
        f = self.client.scatter(data, direct=True, workers=self.workers, keys=[key], external=True)
        while f.status != 'finished' or f is None:
            f = self.client.scatter(data, direct=True, workers=self.workers, keys=[key], external=True)


class deisa_array:
    """
    Deisa virtual array class
    """

    def __init__(self, name, array, selection="All"):
        self.name = name
        self.array = array
        self.selection = selection

    def normalize_slices(self, ls):
        ls_norm = []
        if isinstance(ls, (tuple, list)):
            for s in ls:
                ls_norm.append(self.normalize_slice(s, ls.index(s)))
            return tuple(ls_norm)

    def normalize_slice(self, s, index):
        if s[0] is None:
            s[0] = 0

        if s[1] is None:
            s[1] = self.array.shape[index]

        if s[2] is None:
            s[2] = 1

        elif s[2] < 0:
            raise ValueError(
                f"{s} only positive step values are accepted"
            )
        for i in range(2):
            if s[i] < 0:
                s[i] = self.array.shape[index] + s[i]
        return tuple(s)

    def __getitem__(self, slc):
        selection = []
        default = [None, None, None]
        if isinstance(slc, slice):
            selection.append(self.normalize_slice([slc.start, slc.stop, slc.step], 0))
        elif isinstance(slc, tuple):
            for s in range(len(slc)):
                if isinstance(slc[s], slice):
                    selection.append(self.normalize_slice([slc[s].start, slc[s].stop, slc[s].step], s))
                elif isinstance(slc[s], int):
                    if slc[s] >= 0:
                        selection.append((slc[s], slc[s] + 1, 1))
                    else:
                        selec0 = slc[s] + self.array.shape[s]
                        selec1 = slc[s] + self.array.shape[slc.index(s)] + 1
                        selection.append((selec0, selec1, 1))
                elif isinstance(slc[s], type(Ellipsis)):
                    selection.append(self.normalize_slice([0, None, 1], s))
        elif isinstance(slc, int):
            if slc >= 0:
                selection.append((slc, slc + 1, 1))
            else:
                selection.append((slc + self.array.shape[0], slc + self.array.shape[0] + 1, 1))
        elif isinstance(slc, type(Ellipsis)):
            selection.append((0, self.array.shape[0], 1))
        else:
            raise ValueError(f"{slc} ints, slices and Ellipsis only are supported")
        self.selection = selection
        return self.array.__getitem__(slc)

    def get_name(self):
        return self.name

    def get_array(self):
        return self.array

    def set_selection(self, slc):
        self.selection = slc

    def reset_selection(self):
        self.selection = "All"

    def gc(self):
        del self.array


class deisa_arrays:
    """
    Deisa virtual arrays class
    """

    def __init__(self, arrays):
        self.arrays = []
        self.contract = None
        for name, array in arrays.items():
            self.arrays.append(deisa_array(name, array))

    def __getitem__(self, name):
        for dea in self.arrays:
            if dea.get_name() == name:
                return dea
        raise ValueError(
            f"{name} array does not exist in Deisa data store"
        )

    def add_deisa_array(self, deisa_a, name=None):
        if isinstance(deisa_a, deisa_array):
            self.arrays.append(deisa_a)
        elif isinstance(deisa_a, Array) and isinstance(name, str):
            self.arrays.append(deisa_array(deisa_a, name))

    def get_deisa_array(self, name):
        return self.__getitem__(name)

    def drop_arrays(self, names):
        if isinstance(names, str):
            self.arrays[names].set_selection(None)
        elif isinstance(names, list):
            for a in names:
                self.arrays[a].set_selection(None)

    def reset_contract(self):
        for dea in self.arrays:
            dea.reset_selection()

    def check_contract(self):
        if self.contract is None:
            self.generate_contract()
        return self.contract

    def generate_contract(self):
        self.contract = {}
        for a in self.arrays:
            self.contract[a.name] = a.selection

    def validate_contract(self):
        print("Generated contract", self.contract, flush=True)
        contract = Variable("Contract")
        print("Contract has been signed", flush=True)
        contract.set(self.contract)
        self.gc()

    def gc(self):
        for a in self.arrays:
            a.gc()
        print("Original arrays deleted", flush=True)
