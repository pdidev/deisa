import pytest
from dask.distributed import Client, LocalCluster, Queue
import dask.array as da
from deisa import Deisa
import numpy as np
from deisa import BridgeV1
import dask


@pytest.fixture(scope="session")
def complex_setup():
    """Fixture to set up a Dask LocalCluster with 2 workers."""
    # As a fixture we need:

    # A cluster of two Dask workers
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)

    # two bridges (two MPI workers)
    arrays_description0 = {
        "global_t": {
            "timedim": [0],
            "subsizes": [1, 2, 4],
            "starts": [0, 0, 0],
            "sizes": [5, 4, 4],
        }
    }
    arrays_description1 = {
        "global_t": {
            "timedim": [0],
            "subsizes": [1, 2, 4],
            "starts": [0, 2, 0],
            "sizes": [5, 4, 4],
        }
    }

    arrays_dtype = {"global_t": float}

    b0 = BridgeV1(
        cluster=cluster,
        mpi_rank=0,
        mpi_size=2,
        arrays_description=arrays_description0,
        arrays_description_dtype=arrays_dtype,
    )

    b1 = BridgeV1(
        cluster=cluster,
        mpi_rank=1,
        mpi_size=2,
        arrays_description=arrays_description1,
        arrays_description_dtype=arrays_dtype,
    )

    # A Deisa Adaptor and main client

    adaptor = Deisa(nb_workers=2, cluster=cluster)

    client = adaptor.get_client()

    yield {
        "cluster": cluster,
        "b0": b0,
        "b1": b1,
        "adaptor": adaptor,
        "client": client,
    }

    client.close()
    cluster.close()


@pytest.fixture(scope="session")
def easy_setup():
    """Fixture to set up a Dask LocalCluster with 2 workers."""
    cluster = LocalCluster(n_workers=2, threads_per_worker=2)
    client = Client(cluster)
    workers = list(client.scheduler_info()["workers"].keys())
    w0 = workers[0]
    w1 = workers[1]
    adaptor = Deisa(nb_workers=2, cluster=cluster)

    yield {"cluster": cluster, "client": client, "w0": w0, "w1": w1, "adaptor": adaptor}
    client.close()
    cluster.close()


### Testing basic analytics
# Slowly this should be the place where new incoming analytics should be tested
# A possible idea would be to create a library for HPC "common" analytics which we have tested,
# and are sure they work and can be directly used by the parties of interest.
def test_equality_of_arrays(easy_setup):
    adaptor = easy_setup["adaptor"]
    w0 = easy_setup["w0"]
    w1 = easy_setup["w1"]
    client = easy_setup["client"]

    name = "x"
    shape = (5, 4, 4)
    chunksize = (1, 2, 4)
    dtype = float
    task_to_rank = {(0, 0): "x-rank0", (1, 0): "x-rank1"}

    arr_dsk = adaptor.create_array(name, shape, chunksize, dtype, task_to_rank)

    q0 = Queue("x-rank0")
    q1 = Queue("x-rank1")

    entire_array = []
    for t in range(5):
        data0 = np.random.rand(1, 4, 4)
        data1 = np.random.rand(1, 4, 4)
        entire_array.append((data0, data1))

        f0 = client.scatter(data0, direct=True, workers=[w0])
        f1 = client.scatter(data1, direct=True, workers=[w1])

        q0.put(f0)
        q1.put(f1)
    entire_array = [np.concatenate(pair, axis=1) for pair in entire_array]
    arr_np = np.vstack(entire_array)

    arr_dsk = arr_dsk.compute()

    assert (arr_np == arr_dsk).all()


def test_derivative(easy_setup):

    def derivative(F, dx):
        c0 = 2.0 / 3.0
        dFdx = c0 / dx * (F[3:-1] - F[1:-3] - (F[4:] - F[:-4]) / 8.0)
        return dFdx

    adaptor = easy_setup["adaptor"]
    w0 = easy_setup["w0"]
    w1 = easy_setup["w1"]
    client = easy_setup["client"]

    name = "x"
    shape = (5, 4, 4)
    chunksize = (1, 2, 4)
    dtype = float
    task_to_rank = {(0, 0): "x-rank0", (1, 0): "x-rank1"}

    arr_dsk = adaptor.create_array(name, shape, chunksize, dtype, task_to_rank)

    q0 = Queue("x-rank0")
    q1 = Queue("x-rank1")

    entire_array = []
    for t in range(5):
        data0 = np.random.rand(1, 4, 4)
        data1 = np.random.rand(1, 4, 4)
        entire_array.append((data0, data1))

        f0 = client.scatter(data0, direct=True, workers=[w0])
        f1 = client.scatter(data1, direct=True, workers=[w1])

        q0.put(f0)
        q1.put(f1)

    with dask.config.set(array_optimize=None):
        drv_dsk = derivative(arr_dsk, 1)
        drv_dsk = drv_dsk.compute()

    entire_array = [np.concatenate(pair, axis=1) for pair in entire_array]
    arr_np = np.vstack(entire_array)

    drv_np = derivative(arr_np, 1)

    assert (drv_np == drv_dsk).all()


def test_derivative_mean(easy_setup):

    def derivative(F, dx):
        c0 = 2.0 / 3.0
        dFdx = c0 / dx * (F[3:-1] - F[1:-3] - (F[4:] - F[:-4]) / 8.0)
        return dFdx

    adaptor = easy_setup["adaptor"]
    w0 = easy_setup["w0"]
    w1 = easy_setup["w1"]
    client = easy_setup["client"]

    name = "x"
    shape = (5, 4, 4)
    chunksize = (1, 2, 4)
    dtype = float
    task_to_rank = {(0, 0): "x-rank0", (1, 0): "x-rank1"}

    arr_dsk = adaptor.create_array(name, shape, chunksize, dtype, task_to_rank)

    q0 = Queue("x-rank0")
    q1 = Queue("x-rank1")

    entire_array = []
    for t in range(5):
        data0 = np.random.rand(1, 4, 4)
        data1 = np.random.rand(1, 4, 4)
        entire_array.append((data0, data1))

        f0 = client.scatter(data0, direct=True, workers=[w0])
        f1 = client.scatter(data1, direct=True, workers=[w1])

        q0.put(f0)
        q1.put(f1)

    with dask.config.set(array_optimize=None):
        drv_dsk = derivative(arr_dsk, 1).mean()
        drv_dsk_mean = drv_dsk.compute()

    entire_array = [np.concatenate(pair, axis=1) for pair in entire_array]
    arr_np = np.vstack(entire_array)
    drv_np_mean = derivative(arr_np, 1).mean()

    assert abs(drv_np_mean - drv_dsk_mean) < 0.000000001


# # check for no unwanted data transfer
# def test_criss_cross(easy_setup):
#
#     def derivative(F, dx):
#         c0 = 2.0 / 3.0
#         dFdx = c0 / dx * (F[3:-1] - F[1:-3] - (F[4:] - F[:-4]) / 8.0)
#         return dFdx
#
#     adaptor = easy_setup["adaptor"]
#     w0 = easy_setup["w0"]
#     w1 = easy_setup["w1"]
#     client = easy_setup["client"]
#
#     name = "x"
#     shape = (5, 4, 4)
#     chunksize = (1, 2, 4)
#     dtype = float
#     task_to_rank = {(0, 0): "x-rank0", (1, 0): "x-rank1"}
#
#     arr_dsk = adaptor.create_array(name, shape, chunksize, dtype, task_to_rank)
#
#     q0 = Queue("x-rank0")
#     q1 = Queue("x-rank1")
#
#     entire_array = []
#     for t in range(5):
#         data0 = np.random.rand(1, 4, 4)
#         data1 = np.random.rand(1, 4, 4)
#         entire_array.append((data0, data1))
#
#         f0 = client.scatter(data0, direct=True, workers=[w0])
#         f1 = client.scatter(data1, direct=True, workers=[w1])
#
#         q0.put(f0)
#         q1.put(f1)
#
#     with dask.config.set(array_optimize=None):
#         drv_dsk = derivative(arr_dsk, 1)
#         drv_dsk = drv_dsk.compute()
#
#     entire_array = [np.concatenate(pair, axis=1) for pair in entire_array]
#     arr_np = np.vstack(entire_array)
#
#     drv_np = derivative(arr_np, 1)
#
#     assert (drv_np == drv_dsk).all()
