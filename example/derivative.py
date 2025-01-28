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

from deisa import Deisa, DeisaArrays
from dask.distributed import performance_report, wait
import os
import yaml
import dask

os.environ["DASK_DISTRIBUTED__COMM__UCX__INFINIBAND"] = "True"

# Scheduler file name and configuration file
scheduler_info = 'scheduler.json'
config_file = 'config.yml'
with open(config_file, "r") as f:
    cfg = yaml.safe_load(f)

nb_workers = cfg["workers"]

# Initialize Deisa
adaptor = Deisa(nb_workers = nb_workers, scheduler_file_name = scheduler_info)

# DEISA API

# Get client
client = adaptor.get_client()

arrays: DeisaArrays = adaptor.get_deisa_arrays()

##### API CHANGE
# force user to make more clear per dimension what we are selecting. 
# remove support for [...] which is more prone to bugs.
#####

# Select data - sets self.selection
gt = arrays["global_t"][:, : , :]

# handle contract - contract stays the same
arrays.handle_contract()

# or, for more fine grained control if user needs to dynamically change 
# the contract
# arrays.generate_contract()
# arrays.share_contract()

def Derivee(F, dx):
    """
    First Derivative
       Input: F        = function to be derivate
              dx       = step of the variable for derivative
       Output: dFdx = first derivative of F
    """
    c0 = 2. / 3.
    dFdx = c0 / dx * (F[3: - 1] - F[1: - 3] - (F[4:] - F[:- 4]) / 8.)
    return dFdx


# py-bokeh is needed if you wanna see the perf report
with performance_report(filename="dask-report.html"), dask.config.set(array_optimize=None):

    # Construct a lazy task graph 
    cpt = Derivee(gt, 1).mean()

    # Submit the task graph to the scheduler
    # scheduler gets the graph and doesnt do anything yet.
    s = cpt.compute()

    arrays.gc()

    del gt
    # Print the result, note that "s" is a future object, to get the result of the computation,
    # we call `s.result()` to retreive it.
    print(f"Derivative computation is {s}", flush=True)


print("Done", flush=True)
adaptor.wait_for_last_bridge_and_shutdown()
