import sys
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.linear_model import RidgeCV
import time
# pd.options.mode.chained_assignment = None


LMask_id = np.load('lpp_surface/LMask_fs5_id.npy')

subj_id = int(sys.argv[1])
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
    
])


# load fmri data
fmri_avg = []  # (n_section, n_vertex, n_timepoint_in_section)
fmri_subj = []  # (n_section, n_vertex, n_timepoint_in_section)

for sec_id in range(1, 10):
    # read the fmri data in this section
    fmri_subj_sec = np.load(f'lpp_surface/subjects/{subj}/{subj}-run-{sec_id}-fsaverage7-lh.nii.gz.npy')
    fmri_subj.append(fmri_subj_sec)
    
    fmri_avg_sec = np.load(f'lpp_surface/average/avg-run-{sec_id}-fsaverage7-lh.nii.gz.npy')
    fmri_avg.append(fmri_avg_sec)


# # normalize x
# x = zscore(fmri_avg, nan_policy='omit', axis=-1)  # (n_section, n_vertex, n_timepoint_in_section)
# x = np.nan_to_num(x)  # replace nan with 0
# print(x.shape)


# read subj surface fmri data and do regressions
regression_scores = []  # the r^2 scores for the regression on each vertex
regression_betas = []  # the beta values (1,) for each vertex in the mask (left hemisphere)
n_vertex_in_mask = len(LMask_id)  # 48455 for fs7, 3026 for fs5
# n_vertex_in_mask = 10  # for debugging

start = time.time()
for v in range(n_vertex_in_mask):
    print(f'Vertex {v + 1}/{n_vertex_in_mask}')
    activations_subj = []  # subject fmri activations for this vertex
    activations_avg = []  # average fmri activations for this vertex
    
    # extract the fmri events for this vertex
    for sec_id in range(1, 10):
        df_sec = df_word[df_word['sec_id'] == sec_id]  # section1, ..., section9

        # load the fmri data of this vertex in this section
        fmri_subj_sec = fmri_subj[sec_id - 1]
        fmri_subj_sec_vertex = fmri_subj_sec[LMask_id[v]]  # (n_timepoint_in_section,), different in each section
        # print(fmri_vertex.shape)
        
        fmri_avg_sec = fmri_avg[sec_id - 1]
        fmri_avg_sec_vertex = fmri_avg_sec[LMask_id[v]]  # (n_timepoint_in_section,)
        
        # read the word offsets and the corresponding fmri activation
        for offset in df_sec['sec_offset']:
            timepoint = get_timepoint(offset)
            activation_subj = fmri_subj_sec_vertex[timepoint]  # scalar
            activations_subj.append(activation_subj)
            
            activation_avg = fmri_avg_sec_vertex[timepoint]
            activations_avg.append(activation_avg)  # scalar
            
        
    # normalize the activations as y for this vertex
    x_v = zscore(activations_avg, nan_policy='omit')  # (n_word_total,)
    x_v = np.nan_to_num(x_v)
    x_v = x_v.reshape(-1, 1)
    
    y_v = zscore(activations_subj, nan_policy='omit')  # (n_word_total,)
    y_v = np.nan_to_num(y_v)  # replace nan with 0
    
    # do the ridge regression
    model = RidgeCV()
    model.fit(x_v, y_v)
    
    regression_score = model.score(x_v, y_v)
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

np.save(f'regression_results/intersubject/scores/{subj_id}_scores.npy', regression_scores)
np.save(f'regression_results/intersubject/betas/{subj_id}_betas.npy', regression_betas)


