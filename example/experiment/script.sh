# export DASK_DISTRIBUTED__COMM__UCX__INFINIBAND=True
# export UCX_TLS=ib
echo workingdir $PWD

SCHEFILE=scheduler.json

echo Launching Scheduler... 
dask scheduler --scheduler-file $SCHEFILE 2>scheduler.e&

# Wait for the SCHEFILE to be created 
while ! [ -f $SCHEFILE ]; do
    sleep 3
done

echo Scheduler Launched!

# Launch Dask workers in the rest of the allocated nodes 
echo Launching Dask Workers...

dask worker --scheduler-file $SCHEFILE --local-directory $PWD/workers  --nworkers 2  --nthreads 1 2>workers.e& 

echo Dask Workers Launched!

sleep 5

# Connect the client to the Dask scheduler
echo Connecting Master Client...

`which python` ../derivative.py 2>derivative.e&
# `which python` ../print.py 2>print.e&

client_pid=$!

echo cliend_pid $client_pid

echo Master Client Connected!

sleep 3

echo Running Simulation...
mpirun -n 2 ./simulation 2>simulation.e & 
echo Simulation Finished!
#
# Wait for the client process to be finished 

wait $client_pid
wait

