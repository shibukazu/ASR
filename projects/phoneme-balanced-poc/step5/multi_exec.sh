#!/bin/bash

num_times=$1
command=${@:2}

for i in $(seq 1 $num_times); do
    echo "Running $command"
    $command
done