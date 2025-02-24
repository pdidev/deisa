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

from deisa import Deisa
from dask.distributed import performance_report, wait
import os
import yaml
import dask

os.environ["DASK_DISTRIBUTED__COMM__UCX__INFINIBAND"] = "True"

# Scheduler file name and configuration file
scheduler_info = "scheduler.json"
config_file = "config.yml"
with open(config_file, "r") as f:
    cfg = yaml.safe_load(f)

nb_dask_workers = cfg["workers"]

# Initialize Deisa
adaptor = Deisa(
    nb_expected_dask_workers=nb_dask_workers, scheduler_file_name=scheduler_info
)

# DEISA API

# Get client
client = adaptor.client


def Derivee(F, dx):
    """
    First Derivative
       Input: F        = function to be derivate
              dx       = step of the variable for derivative
       Output: dFdx = first derivative of F
    """
    c0 = 2.0 / 3.0
    dFdx = c0 / dx * (F[3:-1] - F[1:-3] - (F[4:] - F[:-4]) / 8.0)
    return dFdx


# py-bokeh is needed if you wanna see the perf report
with performance_report(filename="dask-report.html"), dask.config.set( # type: ignore
    array_optimize=None
):
    # only 3 chunks needed in dim0, and 1 chunk in dim1
    gt = adaptor["global_t", :, :, :]
    adaptor.ready()
    # print(gt.compute())

    # Construct a lazy task graph
    cpt = Derivee(gt, 1).mean()

    # Submit the task graph to the scheduler
    # scheduler gets the graph and doesnt do anything yet.
    s = cpt.compute()

    # del gt
    # Print the result, note that "s" is a future object, to get the result of the computation,
    # we call `s.result()` to retreive it.
    print(f"Derivative computation is {s}", flush=True)


print("Done", flush=True)
adaptor.wait_for_last_bridge_and_shutdown()
