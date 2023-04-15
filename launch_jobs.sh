#!/bin/bash

configs=(0-0-0-0-0  0-0-1-0-2  0-0-2-1-1  0-1-0-2-0  0-1-1-2-2  0-2-0-0-1  0-2-1-1-0  0-2-2-1-2  1-0-0-2-1  1-0-2-0-0  1-1-0-0-2  1-1-1-1-1  1-1-2-2-0  1-2-0-2-2  1-2-2-0-1  2-0-0-1-0  2-0-1-1-2  2-0-2-2-1  2-1-1-0-0  2-1-2-0-2  2-2-0-1-1  2-2-1-2-0  2-2-2-2-2
0-0-0-0-1  0-0-1-1-0  0-0-2-1-2  0-1-0-2-1  0-1-2-0-0  0-2-0-0-2  0-2-1-1-1  0-2-2-2-0  1-0-0-2-2  1-0-2-0-1  1-1-0-1-0  1-1-1-1-2  1-1-2-2-1  1-2-1-0-0  1-2-2-0-2  2-0-0-1-1  2-0-1-2-0  2-0-2-2-2  2-1-1-0-1  2-1-2-1-0  2-2-0-1-2  2-2-1-2-1
0-0-0-0-2  0-0-1-1-1  0-0-2-2-0  0-1-0-2-2  0-1-2-0-1  0-2-0-1-0  0-2-1-1-2  0-2-2-2-1  1-0-1-0-0  1-0-2-0-2  1-1-0-1-1  1-1-1-2-0  1-1-2-2-2  1-2-1-0-1  1-2-2-1-0  2-0-0-1-2  2-0-1-2-1  2-1-0-0-0  2-1-1-0-2  2-1-2-1-1  2-2-0-2-0  2-2-1-2-2
0-0-0-1-0  0-0-1-1-2  0-0-2-2-1  0-1-1-0-0  0-1-2-0-2  0-2-0-1-1  0-2-1-2-0  0-2-2-2-2  1-0-1-0-1  1-0-2-1-0  1-1-0-1-2  1-1-1-2-1  1-2-0-0-0  1-2-1-0-2  1-2-2-1-1  2-0-0-2-0  2-0-1-2-2  2-1-0-0-1  2-1-1-1-0  2-1-2-1-2  2-2-0-2-1  2-2-2-0-0
0-0-0-1-1  0-0-1-2-0  0-0-2-2-2  0-1-1-0-1  0-1-2-1-0  0-2-0-1-2  0-2-1-2-1  1-0-0-0-0  1-0-1-0-2  1-0-2-1-1  1-1-0-2-0  1-1-1-2-2  1-2-0-0-1  1-2-1-1-0  1-2-2-1-2  2-0-0-2-1  2-0-2-0-0  2-1-0-0-2  2-1-1-1-1  2-1-2-2-0  2-2-0-2-2  2-2-2-0-1
0-0-0-1-2  0-0-1-2-1  0-1-0-0-0  0-1-1-0-2  0-1-2-1-1  0-2-0-2-0  0-2-1-2-2  1-0-0-0-1  1-0-1-1-0  1-0-2-1-2  1-1-0-2-1  1-1-2-0-0  1-2-0-0-2  1-2-1-1-1  1-2-2-2-0  2-0-0-2-2  2-0-2-0-1  2-1-0-1-0  2-1-1-1-2  2-1-2-2-1  2-2-1-0-0  2-2-2-0-2
0-0-0-2-0  0-0-1-2-2  0-1-0-0-1  0-1-1-1-0  0-1-2-1-2  0-2-0-2-1  0-2-2-0-0  1-0-0-0-2  1-0-1-1-1  1-0-2-2-0  1-1-0-2-2  1-1-2-0-1  1-2-0-1-0  1-2-1-1-2  1-2-2-2-1  2-0-1-0-0  2-0-2-0-2  2-1-0-1-1  2-1-1-2-0  2-1-2-2-2  2-2-1-0-1  2-2-2-1-0
0-0-0-2-1  0-0-2-0-0  0-1-0-0-2  0-1-1-1-1  0-1-2-2-0  0-2-0-2-2  0-2-2-0-1  1-0-0-1-0  1-0-1-1-2  1-0-2-2-1  1-1-1-0-0  1-1-2-0-2  1-2-0-1-1  1-2-1-2-0  1-2-2-2-2  2-0-1-0-1  2-0-2-1-0  2-1-0-1-2  2-1-1-2-1  2-2-0-0-0  2-2-1-0-2  2-2-2-1-1
0-0-0-2-2  0-0-2-0-1  0-1-0-1-0  0-1-1-1-2  0-1-2-2-1  0-2-1-0-0  0-2-2-0-2  1-0-0-1-1  1-0-1-2-0  1-0-2-2-2  1-1-1-0-1  1-1-2-1-0  1-2-0-1-2  1-2-1-2-1  2-0-0-0-0  2-0-1-0-2  2-0-2-1-1  2-1-0-2-0  2-1-1-2-2  2-2-0-0-1  2-2-1-1-0  2-2-2-1-2
0-0-1-0-0  0-0-2-0-2  0-1-0-1-1  0-1-1-2-0  0-1-2-2-2  0-2-1-0-1  0-2-2-1-0  1-0-0-1-2  1-0-1-2-1  1-1-0-0-0  1-1-1-0-2  1-1-2-1-1  1-2-0-2-0  1-2-1-2-2  2-0-0-0-1  2-0-1-1-0  2-0-2-1-2  2-1-0-2-1  2-1-2-0-0  2-2-0-0-2  2-2-1-1-1  2-2-2-2-0
0-0-1-0-1  0-0-2-1-0  0-1-0-1-2  0-1-1-2-1  0-2-0-0-0  0-2-1-0-2  0-2-2-1-1  1-0-0-2-0  1-0-1-2-2  1-1-0-0-1  1-1-1-1-0  1-1-2-1-2  1-2-0-2-1  1-2-2-0-0  2-0-0-0-2  2-0-1-1-1  2-0-2-2-0  2-1-0-2-2  2-1-2-0-1  2-2-0-1-0  2-2-1-1-2  2-2-2-2-1)

for config in "${configs[@]}"; do
    ctl job run --name "jtien-$config-job" --command "cd nl_pref/robosuite-benchmark/ && export PYTHONPATH=.:$PYTHONPATH && python scripts/train.py --variant training_configs/LiftModded-Jaco-OSC-POSITION-SEED251/diverse-rewards/$config/variant.json --seed 251 --log_dir ../log/runs/diverse-rewards/$config/" \
                --container=docker.io/jeremytien/devbox:robosuite-benchmark \
                --gpu 1 --cpu 8 \
                --shared-host-dir-mount /code \
                --force-pull \
                --volume-name ${USER}-home \
                --volume-mount /${USER}-data
done