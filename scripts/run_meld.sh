#!/bin/bash

# Run the benchmarking script in the background using nohup
nohup python calculate_meld.py \
  --llms gemini \
  --benchmarks medagentsbench medarc metamedqa \
  > meld_results.txt 2>&1 &