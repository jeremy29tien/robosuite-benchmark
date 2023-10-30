Description of data.

traj_a_indexes.npy and traj_b_indexes.npy are the trajectory portion of the dataset.
nlcomp_indexes.npy is the corresponding language portion of the dataset.
in the training set, both are 5733000 length arrays, which is composed of
- 196 unique trajectories
- and thus 196 choose 2 = 19110 pairs of trajectories
- each (trajectory pair) of which are compared on 5 features (gt reward, speed, height, distance to cube, distance to bottle)
- each (feature) of which have 3 synonymous adjectives
- each (adjective) of which has an antonym (opposite)

hence, we have 19110 * 5 features * 3 adjectives * 2 (for opposites) = 573300 datapoints

then, we perform the noisy data augmentation -- for each datapoint, we duplicate it 10 times and flip the label for each with probability equal to the sigmoid of the feature difference. this gives us the new training dataset size of 5733000. 

the validation set is constructed similarly -- 22 unique trajectories (separate from the 196 in the training dataset) and thus 231 unique pairs. the augmentation for 5 features, 3 adjectives, and 2 opposites is similarly performed, but the noisy data augmentation is not. hence, the size is 6930. 

the actual trajectories are located in trajs.npy (of which traj_a_indexes.npy and traj_b_indexes.npy are indexes). similarly, the BERT-encoded form of the actual language commands (labels)  are in unique_nlcomps.npy (of which nlcomp_indexes.npy is an index).

unique_nlcomps_for_aprel.* are files that contain the commands in natural language (.json) and after BERT preprocessing (.npy) that are used for running active learning (APReL); these were generated after the rest of the dataset. 


