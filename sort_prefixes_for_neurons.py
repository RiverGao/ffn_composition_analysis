import numpy as np


dataset = 'reading_brain'

with open(f'act_out/{dataset}/prefixes.txt') as f_pref:
    prefixes = f_pref.read().strip().split('\n')

n_layer = 32
d_hidden = 11008

for lyr in range(n_layer):
    layer_activations = np.load(f'act_out/{dataset}/act_layer{lyr}.npy')

    with open(f'sortings_activation/{dataset}/layer{lyr}.txt', 'w') as f_sort:
        for d in range(d_hidden):
            neuron_activations = layer_activations[:, d]
            sort_idx = np.argsort(neuron_activations)
            sorted_prefixes = np.take(prefixes, sort_idx)[:25]

            f_sort.write(f'## neuron {d}\n')
            f_sort.write('\n'.join(sorted_prefixes) + '\n\n')
