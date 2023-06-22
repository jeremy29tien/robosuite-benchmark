#!/bin/bash

configs=(0-0-0-0-2  0-0-1-0-2  0-0-2-1-1  0-1-0-0-1  0-1-1-1-1  0-1-2-2-0  0-2-0-2-1  0-2-2-0-0  1-0-0-2-1  1-0-1-1-1  1-0-2-2-0  1-1-0-2-1  1-1-2-0-0  1-1-2-2-1  1-2-1-1-1  2-0-0-2-2  2-1-0-1-0  2-1-1-1-2  2-1-2-2-1  2-2-1-0-0  2-2-2-0-2
0-0-0-2-0  0-0-1-1-0  0-0-2-1-2  0-1-0-2-1  0-1-1-2-2  0-2-0-0-2  0-2-0-2-2  1-0-0-0-0  1-0-0-2-2  1-0-2-0-0  1-1-0-1-0  1-1-1-0-0  1-1-2-0-1  1-2-0-0-1  1-2-1-1-2  2-0-1-2-0  2-1-0-1-1  2-1-1-2-0  2-2-0-1-1  2-2-1-0-1  2-2-2-1-0
0-0-0-2-1  0-0-1-2-2  0-0-2-2-2  0-1-1-1-0  0-1-2-0-0  0-2-0-1-0  0-2-1-1-2  1-0-0-0-2  1-0-1-0-2  1-0-2-1-1  1-1-0-2-0  1-1-1-1-2  1-1-2-2-0  1-2-0-0-2  1-2-2-2-0  2-0-2-2-2  2-1-1-0-1  2-1-2-0-2  2-2-0-1-2  2-2-1-2-0
)

for config in "${configs[@]}"; do
    python scripts/rollout.py --load_dir "log/runs/diverse-rewards/$config/LiftModded-Jaco-OSC-POSITION-SEED251/*/" \
    --camera "frontview" --record_video_per_rollout --seed 251 --num_episodes 3 \
    --output_dir "data/diverse-rewards/$config/with_camera_obs" \
    --video_output_dir "data/diverse-rewards/videos"
done
