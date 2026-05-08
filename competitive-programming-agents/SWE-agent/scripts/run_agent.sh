#!/bin/bash

# Given total cost
cost=$1


for d in 1 3 5
do
    budget=$(echo "$cost / $d" | bc -l)   # bc does floating-point division
    echo "$cost divided by $d = $budget"
    sweagent run-batch \
  --instances.type=file\
  --instances.path data/cf_swebench_style.json \
  --config config/cp_claude.yaml\
  --instances.slice :2 \
  --instances.deployment.python_standalone_dir="" \
  --progress_bar=False \
  --agent.model.name=claude-sonnet-4-20250514 \
  --agent.model.per_instance_cost_limit=$budget
done
