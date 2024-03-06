import numpy as np


n_layer = 32

for section_id in range(1, 10):
    for model_version in ['chat', 'base']:
        for lyr in range(n_layer):
            comp_scores = np.load(f'scores_composition/lpp_en/section{section_id}/{model_version}_layer{lyr}.npy')  # (n_prefix, )
            print(comp_scores.shape)
            
            np.random.shuffle(comp_scores)
            np.save(f'scores_composition/lpp_en/section{section_id}/{model_version}_shuffled_layer{lyr}.npy', comp_scores)