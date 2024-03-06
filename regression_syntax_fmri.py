import sys
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import RidgeCV
import time
# pd.options.mode.chained_assignment = None


LMask_id = np.load('lpp_surface/LMask_fs5_id.npy')

tree_type = sys.argv[1]
subj_id = int(sys.argv[2])

# tree_type = 'bu'
# subj_id = 1

subj = f'sub-EN00{subj_id}' if subj_id < 10 else f'sub-EN0{subj_id}'


# function to extract fmri timepoint from word offset
def get_timepoint(time_offset, delay=5000, dur=2000):
    point = np.ceil((time_offset + delay) / dur)
    return int(point)


# word information
df_word = pd.read_csv('sentences/lpp_en_word_info.csv', usecols=[
    'lemma',
    'sec_id',
    'sec_onset',
    'sec_offset',
    'pref_id',
    f'depth_{tree_type}'
])


# load model composition scores (x) and fmri data
x = []  # (n_word_total, n_layer)
fmri_data = []  # (n_section, n_vertex_in_mask, n_timepoint_in_section)

for sec_id in range(1, 10):
    df_sec = df_word[df_word['sec_id'] == sec_id]  # section1, ..., section9
    n_word = len(df_sec)
    
    # read the syntactic complexity scores
    scores_arrays = [df_sec[f'depth_{tree_type}'].to_numpy()]
    
    # add a section-start label due to the feature of fMRI
    section_start_label = np.zeros_like(scores_arrays[-1])  # value=1 at the start of each section, otherwise 0
    section_start_label[0] = 1
    scores_arrays.append(section_start_label)  # n_word_in_sec
    
    x.append(np.stack(scores_arrays, axis=0).T)  # (n_word_in_sec, 2)
    
    # read the fmri data in this section
    fmri_sec = np.load(f'lpp_surface/subjects/{subj}/{subj}-run-{sec_id}-fsaverage7-lh.nii.gz.npy')
    fmri_data.append(fmri_sec)
    
x = np.concatenate(x, axis=0) # (n_word_total, 2)


# normalize x
x = zscore(x, nan_policy='omit')  # (n_word_total, 2), already an array
x = np.nan_to_num(x)  # replace nan with 0
print(x.shape)

# read subj surface fmri data and do regressions
regression_scores = []  # the r^2 scores for the regression on each vertex
regression_betas = []  # the beta values (n_layer-dim) for each vertex in the mask (left hemisphere)
n_vertex_in_mask = len(LMask_id)  # 48455 for fs7, 3026 for fs5
# n_vertex_in_mask = 10  # for debugging

start = time.time()
for v in range(n_vertex_in_mask):
    print(f'Vertex {v + 1}/{n_vertex_in_mask}')
    activations = []  # fmri activations for this vertex
    
    # extract the fmri events for this vertex
    for sec_id in range(1, 10):
        df_sec = df_word[df_word['sec_id'] == sec_id]  # section1, ..., section9
        n_word = len(df_sec)
        
        # load the fmri data of this vertex in this section
        fmri_sec = fmri_data[sec_id - 1]
        fmri_vertex = fmri_sec[LMask_id[v]]  # (n_timepoint_in_section,), different in each section
        # print(fmri_vertex.shape)
        
        # read the word offsets and the corresponding fmri activation
        for offset in df_sec['sec_offset']:
            timepoint = get_timepoint(offset)
            activation = fmri_vertex[timepoint]
            activations.append(activation)
        
    # normalize the activations as y for this vertex
    y_v = zscore(activations, nan_policy='omit')  # (n_word_total,)
    y_v = np.nan_to_num(y_v)  # replace nan with 0
    
    # do the ridge regression
    model = RidgeCV()
    model.fit(x, y_v)
    
    regression_score = model.score(x, y_v)
    print(f'\tRegression score: {regression_score}')
    regression_scores.append(regression_score)
    
    regression_betas.append(model.coef_[:-1])  # drop the section_start_label
    
    end = time.time()
    elapsed = end - start
    print(f'\tTotal time taken: {elapsed:.6f} seconds')

end = time.time()
elapsed = end - start
print(f'Total time taken: {elapsed:.6f} seconds')

regression_scores = np.array(regression_scores)  # (n_vertex_in_mask,)
regression_betas = np.array(regression_betas)  # (n_vertex_in_mask, 1)

regression_betas[regression_scores == 1] *= 0  # regression_score == 1 => all-nan vertex
regression_scores[regression_scores == 1] = 0  # note: must handle betas before scores

np.save(f'regression_results/syntax-fmri/scores/{tree_type}_{subj_id}_scores.npy', regression_scores)
np.save(f'regression_results/syntax-fmri/betas/{tree_type}_{subj_id}_betas.npy', regression_betas)


