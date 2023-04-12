import numpy as np
import os
import json


for gt in range(3):
    gt_weight_low = gt * 2 / 3 - 1
    gt_weight_high = gt_weight_low + 2 / 3
    gt_weight = np.random.uniform(low=gt_weight_low, high=gt_weight_high)

    for speed in range(3):
        speed_weight_low = speed * 2 / 3 - 1
        speed_weight_high = speed_weight_low + 2 / 3
        speed_weight = np.random.uniform(low=speed_weight_low, high=speed_weight_high)

        for height in range(3):
            height_weight_low = height * 2 / 3 - 1
            height_weight_high = height_weight_low + 2 / 3
            height_weight = np.random.uniform(low=height_weight_low, high=height_weight_high)

            for distance_to_bottle in range(3):
                distance_to_bottle_weight_low = distance_to_bottle * 2 / 3 - 1
                distance_to_bottle_weight_high = distance_to_bottle_weight_low + 2 / 3
                distance_to_bottle_weight = np.random.uniform(low=distance_to_bottle_weight_low,
                                                              high=distance_to_bottle_weight_high)

                for distance_to_cube in range(3):
                    distance_to_cube_weight_low = distance_to_cube * 2 / 3 - 1
                    distance_to_cube_weight_high = distance_to_cube_weight_low + 2 / 3
                    distance_to_cube_weight = np.random.uniform(low=distance_to_cube_weight_low,
                                                                high=distance_to_cube_weight_high)

                    # TODO: deepcopy of template for efficiency?? orrr just read in template every time.
                    #  Simplest just to read in template every time.
                    template_path = "training_configs/LiftModded-Jaco-OSC-POSITION-SEED251/diverse-rewards/template/"
                    filename = "variant.json"

                    # Opening JSON file
                    with open(os.path.join(template_path, filename), 'r') as openfile:
                        template = json.load(openfile)

                    template["eval_environment_kwargs"]["weights"] = [gt_weight, speed_weight, height_weight,
                                                                      distance_to_bottle_weight,
                                                                      distance_to_cube_weight]
                    template["expl_environment_kwargs"]["weights"] = [gt_weight, speed_weight, height_weight,
                                                                      distance_to_bottle_weight,
                                                                      distance_to_cube_weight]

                    output_path = "training_configs/LiftModded-Jaco-OSC-POSITION-SEED251/diverse-rewards/" \
                                  "{}-{}-{}-{}-{}/".format(gt, speed, height, distance_to_bottle, distance_to_cube)
                    if not os.path.exists(output_path):
                        os.makedirs(output_path)
                    with open(os.path.join(output_path, filename), 'wb') as temp_file:
                        json.dump(template, temp_file)
