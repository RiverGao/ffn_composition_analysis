import numpy as np


d_prefix = {
    'model-fmri': ['base', 'chat', 'base_shuffled', 'chat_shuffled', 'random'],
    'syntax-fmri': ['bu', 'td', 'lc', 'logfreq']
}

intersubj_scores = []
for j_subj in range(1, 50):
    intersubj_score_subj = np.load(f'regression_results/intersubject/scores/{j_subj}_scores.npy')  # (n_vetex_in_mask,)
    intersubj_scores.append(intersubj_score_subj)
    
avg_intersubj_scores = np.mean(intersubj_scores, axis=0)  # (n_vetex_in_mask,)
print(avg_intersubj_scores.shape)

for regression_name in d_prefix.keys():
    file_prefixes = d_prefix[regression_name]
    
    for prefix in file_prefixes:
        regression_scores = []
        
        for j_subj in range(1, 50):
            regression_score_subj = np.load(f'regression_results/{regression_name}/scores/{prefix}_{j_subj}_scores.npy')  # (n_vetex_in_mask,)
            regression_scores.append(regression_score_subj)
            
        avg_regression_scores = np.mean(regression_scores, axis=0)  # (n_vetex_in_mask,)
        print(f'{regression_name}, {prefix}: {avg_regression_scores.shape}')
        
        rescaled_avg_regression_scores = avg_regression_scores / avg_intersubj_scores
        print(f'\tMax score: {max(rescaled_avg_regression_scores):.6f};  Mean score: {np.mean(rescaled_avg_regression_scores):.6f}')
        

# (3026,)
# model-fmri, base: (3026,)
#         Max score: 0.177391;  Mean score: 0.060273
# model-fmri, chat: (3026,)
#         Max score: 0.136077;  Mean score: 0.046163
# model-fmri, base_shuffled: (3026,)
#         Max score: 0.219284;  Mean score: 0.073376
# model-fmri, chat_shuffled: (3026,)
#         Max score: 0.182591;  Mean score: 0.059317
# model-fmri, random: (3026,)
#         Max score: 0.069710;  Mean score: 0.022879
# syntax-fmri, bu: (3026,)
#         Max score: 0.000521;  Mean score: 0.000157
# syntax-fmri, td: (3026,)
#         Max score: 0.003668;  Mean score: 0.001099
# syntax-fmri, lc: (3026,)
#         Max score: 0.006358;  Mean score: 0.001827
# syntax-fmri, logfreq: (3026,)
#         Max score: 0.006735;  Mean score: 0.001984
