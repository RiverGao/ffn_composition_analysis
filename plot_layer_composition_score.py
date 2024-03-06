import numpy as np
from matplotlib import pyplot as plt


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

avg_scores_base = []
avg_scores_chat = []
avg_scores_random = []
for lyr in range(32):
    layer_scores_base = []
    layer_scores_chat = []
    layer_scores_random = []
    for sec in range(1, 10):
        layer_scores_base.extend(np.load(f'scores_composition/lpp_en/section{sec}/base_layer{lyr}.npy').tolist())
        layer_scores_chat.extend(np.load(f'scores_composition/lpp_en/section{sec}/chat_layer{lyr}.npy').tolist())
        layer_scores_random.extend(np.load(f'scores_composition/lpp_en/section{sec}/random_layer{lyr}.npy').tolist())
        
    avg_scores_base.append(np.mean(layer_scores_base))
    avg_scores_chat.append(np.mean(layer_scores_chat))
    avg_scores_random.append(np.mean(layer_scores_random))


# plot the average composition scores
x_labels = [str(i) for i in range(1, 33)]  # layer numbers as the x labels
x_loc = np.arange(len(x_labels))  # the label locations
width = 0.13  # the width of the bars
multiplier = 0

fig, ax = plt.subplots()
fig.set_size_inches(14, 4)

target_labels = ['LLaMA2-base', 'LLaMA2-chat', 'random']
colors = ['xkcd:dull orange', 'xkcd:dusky rose', 'xkcd:grey']
avg_scores = [avg_scores_base, avg_scores_chat, avg_scores_random]
for i_tar, l_tar in enumerate(target_labels):
    offset = width * multiplier
    rects = ax.bar(x_loc + offset, avg_scores[i_tar], width, label=l_tar, color=colors[i_tar])
    # ax.bar_label(rects, padding=3)
    multiplier += 1
    
# avg_scores = [avg_scores_base, avg_scores_chat, avg_scores_random]
# for i_tar, l_tar in enumerate(target_labels):
#     line = ax.plot(x_loc, avg_scores[i_tar], label=l_tar)
#     # ax.bar_label(rects, padding=3)
#     multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Composition Score')
ax.set_xlabel('Model Layer')
ax.set_title(f'Layerwise average Composition Scores', fontweight='bold')
ax.set_xticks(x_loc + width, x_labels)
ax.legend(ncols=3)
ax.set_ylim(0.6, 1.05)

plt.tight_layout()
plt.savefig(f'figures/model_layer_composition_score.png', dpi=120)

