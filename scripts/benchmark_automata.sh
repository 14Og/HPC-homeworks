#!/bin/bash

OUTPUT_CSV="assets/speedup.csv"
echo "Processes,Time_Sec" > $OUTPUT_CSV

# Parameters for benchmarking (needs to be large enough to see speedup)
CELLS=100000
STEPS=10000
RULE=110

echo "Starting Benchmark: Rule $RULE, Cells $CELLS, Steps $STEPS"
echo "--------------------------------------------------------"

for p in {1..16}; do
    echo -n "Running with $p processes... "
    # Run and capture output. Notice the last argument is '0' so it doesn't print the grid.
    # --oversubscribe is used safely in case you run more processes than physical cores
    OUTPUT=$(mpirun -n $p --oversubscribe ./build/Automata $RULE --static $CELLS $STEPS 0)
    
    # Extract the time using grep and awk
    TIME=$(echo "$OUTPUT" | grep "Total Simulation Time" | awk '{print $4}')
    
    echo "${TIME}s"
    echo "$p,$TIME" >> $OUTPUT_CSV
done

echo "--------------------------------------------------------"
echo "Benchmarking complete"
