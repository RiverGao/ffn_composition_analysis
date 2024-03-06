from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import zscore


n_layer = 32
model_version = 'chat'

df_word = pd.read_csv('sentences/lpp_en_word_info.csv', usecols=[
    'lemma',
    'sec_id',
    'logfreq',
    'pref_id',
    'depth_bu',
    'depth_lc',
    'depth_td'
])

# do one regression with the data from all 9 sections
regression_scores = [0, 0, 0, 0]  # each element is a scalar
regression_betas = [None, None, None, None]  # each element is a (n_layer,) ndarray

y_logfreq = []  # (n_word_total,), same as below
y_bu = []
y_lc = []
y_td = []

x = []  # (n_word_total, n_layer)

for sec_id in range(1, 10):
    df_sec = df_word[df_word['sec_id'] == sec_id]  # section1, ..., section9
    n_word = len(df_sec)
    
    y_logfreq.extend(df_sec['logfreq'].tolist())  # (n_word_section,), same as below
    y_bu.extend(df_sec['depth_bu'].tolist())
    y_lc.extend(df_sec['depth_lc'].tolist())
    y_td.extend(df_sec['depth_td'].tolist())
    
    # read the composition scores
    scores_arrays = []  # to stack composition scores from all the layers
    for lyr in range(n_layer):
        scores_arrays.append(np.load(f'scores_composition/lpp_en/section{sec_id}/{model_version}_layer{lyr}.npy'))  # n_prefix
    # scores_arrays.append(np.ones_like(scores_arrays[-1]) * sec_id)  # n_perfix
    compo_scores = np.stack(scores_arrays, axis=0)  # (n_layer, n_prefix)
    
    for pref_id in df_sec['pref_id']:
        x.append(compo_scores[:, pref_id])  # (n_layer,)


# normalize the x and y data
x = zscore(x)  # (n_word_total, n_layer), already an array
y_logfreq = zscore(y_logfreq)
y_bu = zscore(y_bu)
y_lc = zscore(y_lc)
y_td = zscore(y_td)

# calculate the covariance of x
corrx = np.corrcoef(x.T)
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
# corr_mat = ax.imshow(abs(corrx), vmin=0, vmax=1)
corr_mat = ax.imshow(corrx)

ax.set_ylabel('Model Layer')
ax.set_xlabel('Model Layer')
ax.set_title('Pearson\'s r of Layerwise Composition Scores')
plt.colorbar(corr_mat)

plt.savefig(f'figures/{model_version}_corrx_raw.png', dpi=120)


# build the ridge regression models
model = Ridge(alpha=1.0)

model.fit(x, y_logfreq)
regression_scores[0] = model.score(x, y_logfreq)
# regression_betas[0] = abs(model.coef_)
regression_betas[0] = model.coef_

model.fit(x, y_bu)
regression_scores[1] = model.score(x, y_bu)
# regression_betas[1] = abs(model.coef_)
regression_betas[1] = model.coef_

model.fit(x, y_lc)
regression_scores[2] = model.score(x, y_lc)
# regression_betas[2] = abs(model.coef_)
regression_betas[2] = model.coef_

model.fit(x, y_td)
regression_scores[3] = model.score(x, y_td)
# regression_betas[3] = abs(model.coef_)
regression_betas[3] = model.coef_

d_regression = {
    # 'Section':[str(i) for i in range(1, 10)],
    'Log Frequency': regression_scores[0], 
    'Bottom-Up': regression_scores[1], 
    'Left-Corner': regression_scores[2], 
    'Top-Down': regression_scores[3]
}
print(d_regression)

# df_regression = pd.DataFrame(d_regression)
# mean_scores = df_regression.mean(numeric_only=True).to_frame().T
# mean_scores.index = [len(df_regression)]
# df_regression = pd.concat([df_regression, mean_scores])
# df_regression.iloc[-1, 0] = 'mean'

# df_regression.to_csv('regression_results/model_control/results_control_all.csv', index=False)


# plot the beta values
x_labels = [str(i) for i in range(32)]  # layer numbers as the x labels
x_loc = np.arange(len(x_labels))  # the label locations
width = 0.13  # the width of the bars
multiplier = 0

fig, ax = plt.subplots()
fig.set_size_inches(16, 4)

target_labels = ['Log Frequency', 'Bottom-Up Parsing', 'Left-Corner Parsing', 'Top-Down Parsing']
for i_tar, l_tar in enumerate(target_labels):
    offset = width * multiplier
    rects = ax.bar(x_loc + offset, regression_betas[i_tar], width, label=l_tar)
    # ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Beta Value')
ax.set_xlabel('Model Layer')
ax.set_title('Layerwise Regression Beta Values with the Controlling Variables')
ax.set_xticks(x_loc + width, x_labels)
ax.legend(loc='upper left', ncols=4)
# ax.set_ylim(0, 250)

plt.savefig(f'figures/{model_version}_model_control_betas_raw.png', dpi=120)