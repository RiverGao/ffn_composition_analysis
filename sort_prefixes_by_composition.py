import numpy as np
import sys


model_ver = 'base'  # base, chat, random
dataset = 'lpp_en'
section_num = eval(sys.argv[1])
n_layer = 32

cscores = []  # composition scores

with open(f'act_out/{dataset}/prefixes.txt', 'r') as f:
    sections = f.read().strip().split('\n\n')
    prefixes = sections[section_num - 1].strip().split('\n')
n_prefix = len(prefixes)

for lyr in range(n_layer):
    layer_scores = np.load(f'scores_composition/{dataset}/section{section_num}/{model_ver}_layer{lyr}.npy')  # (n_prefix,)
    assert len(layer_scores) == n_prefix
    cscores.append(layer_scores)

    # select the first and last 50 prefixes w.r.t composition scores
    sort_idx = np.argsort(layer_scores)
    first50_idx = sort_idx[:50]  # the smallest 50 composition scores
    last50_idx = sort_idx[-50:][::-1]  # the largest 50 composition scores

    first50_pref = np.take(prefixes, first50_idx)
    first50_scores = np.take(layer_scores, first50_idx)

    last50_pref = np.take(prefixes, last50_idx)
    last50_scores = np.take(layer_scores, last50_idx)

    with open(f'sortings_composition/{dataset}/section{section_num}/{model_ver}_layer{lyr}.txt', 'w') as f:
        f.write('## 50 prefixes with the lowest composition scores:\n')
        for pref, score in zip(first50_pref, first50_scores):
            f.write(f'{pref} || {score}\n')
        f.write('\n\n')

        f.write('## 50 prefixes with the highest comosition scores:\n')
        for pref, score in zip(last50_pref, last50_scores):
            f.write(f'{pref} || {score}\n')
        f.write('\n')

all_layer_scores = np.stack(cscores, axis=0)  # (n_layer, n_prefix)?
assert all_layer_scores.shape[0] == n_layer and all_layer_scores.shape[1] == n_prefix

overall_scores = np.mean(all_layer_scores, axis=0)  # minimum composition score over all layers
# Why minumum? 
# Because if the prediction is not composed somewhere, 
# i.e. encoded by some single neuron, then it is not composed in the whole picture.
np.save(f'scores_composition/{dataset}/section{section_num}/{model_ver}_all_layers.npy', overall_scores)  # (n_prefix,)

sort_idx = np.argsort(overall_scores)  # composition score small --> large
first50_idx = sort_idx[:50]
last50_idx = sort_idx[-50:][::-1]

first50_pref = np.take(prefixes, first50_idx)
first50_scores = np.take(overall_scores, first50_idx)

last50_pref = np.take(prefixes, last50_idx)
last50_scores = np.take(overall_scores, last50_idx)

with open(f'sortings_composition/{dataset}/section{section_num}/{model_ver}_all_layers.txt', 'w') as f:
    f.write('## 50 prefixes with the lowest composition scores:\n')
    for pref, score in zip(first50_pref, first50_scores):
        f.write(f'{pref} || {score}\n')
    f.write('\n\n')

    f.write('## 50 prefixes with the highest comosition scores:\n')
    for pref, score in zip(last50_pref, last50_scores):
        f.write(f'{pref} || {score}\n')
    f.write('\n')