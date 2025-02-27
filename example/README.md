# Executing the example locally 

After cloning the repo, with `git switch support_official` and switching to the proper branch `git switch support_official`, it is important to setup correctly the environment. To do so we recommend to use NIX + direnv.

To run the example locally follow those steps:

1. Enter the proper directory:
   `cd example`
2. Compile the code:
   ```
   cmake -S . -B build
   make -C build
   ```
3. Copy the simulation to the directory `experiment`:
   `cp build/simulation experiment/`
4. Enter in directory `experiment`:
   `cd experiment`
5. Execute the script `script.sh`, this will launch a Dask scheduler, launch the Dask workers, execute the python code in `../derivative.py`, and finally run the simulation:
   `bash script.sh`

Note: after running it, if you finish the script with Ctrl+C it is important to kill Dask processes: `pkill dask`.