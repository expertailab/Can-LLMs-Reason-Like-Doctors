#!/bin/bash

# Run the benchmarking script in the background using nohup
nohup python benchmarking.py \
  --llms gemini \
  --benchmarks medagentsbench medarc metamedqa \
  --prompts base \
  --seeds 0 \
  > benchmark_results.txt 2>&1 &