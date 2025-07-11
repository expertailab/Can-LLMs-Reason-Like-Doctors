#!/bin/bash

# Run the benchmarking script in the background using nohup
nohup python evaluating.py \
  --llms gemini \
  --benchmarks medagentsbench medarc metamedqa \
  > evaluating_results.txt 2>&1 &