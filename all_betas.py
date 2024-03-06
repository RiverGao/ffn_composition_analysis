import numpy as np
from scipy.stats import zscore

# read regression betas
regression_name = 'model-fmri'
model_version = 'chat_shuffled'
n_subject = 49


betas_list = []
for j in range(1, n_subject + 1):
    beta = np.load(f'regression_results/{regression_name}/betas/{model_version}_{j}_betas.npy')  # (n_vertice, n_layer)
    print(beta.shape)
    
    # beta = zscore(beta, axis=0)
    betas_list.append(beta)
    
data = np.stack(betas_list, axis=0)  # (n_subject, n_vetice, n_layer), n_layer is viewed as n_time
print(data.shape)

np.save(f'regression_results/{regression_name}/betas/{model_version}_all_betas.npy', data)
