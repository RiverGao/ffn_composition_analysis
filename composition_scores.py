import numpy as np
import torch
from torch.nn.functional import kl_div
import sys
from scipy.stats import sem
from scipy.special import rel_entr
from transformers import LlamaForCausalLM, LlamaConfig
from accelerate import dispatch_model, infer_auto_device_map


model_ver = sys.argv[1]  # base, chat, random
model_name = 'Llama-2-7b-chat-hf' if model_ver == 'chat' else 'Llama-2-7b-hf'
model_path = f'/home/nfs02/model/llama2/hf/{model_name}'
# model_path = f'/home/nfs01/gaocj/chinese-alpaca-{model_size}/'
dataset = 'lpp_en'

section_num = eval(sys.argv[2])
lyr_start = 0
lyr_end = 31

# model initialization
if model_ver == 'random':
    config = LlamaConfig(load_in_8bit=False)
    model = LlamaForCausalLM(config)
    model = model.to("cuda")
else:
    model = LlamaForCausalLM.from_pretrained(
        model_path, 
        device_map= "auto", 
        load_in_8bit=False)
model.eval()

n_layer = model.config.num_hidden_layers
d_hidden = model.config.intermediate_size
d_model = model.config.hidden_size
n_vocab = model.config.vocab_size
k_roi = 3000

e_tensor = model.state_dict()['lm_head.weight'].data  # shape of E: n_vocab * d_model
v_tensors = []
for name, param in model.named_parameters():
    if 'mlp.down_proj' in name:
        i_layer = eval(name.split('.')[2])  # "model.layers.0.mlp.down_proj.weight"
        if lyr_start <= i_layer <= lyr_end:
            v_tensors.append(param.data)  # shape of V: d_model * d_hidden
        else:
            v_tensors.append(0)  # placeholder

del model

def entropy(x):
    log_x = torch.log2(x)
    ent = -torch.sum(x * log_x)
    return ent

def entropy_batch(x):
    # x: (bsz, n_vocab)
    log_x = torch.log2(x)
    batch_ent = -torch.sum(x * log_x, 1)
    return batch_ent  # (bsz,)

def variation_of_information(d_x, d_y):
    # Calculate the joint distribution
    joint_distribution = torch.outer(d_x, d_y)  # (32000, 32000)
    
    # Calculate the entropy of the marginal distributions
    entropy_x = entropy(d_x)
    entropy_y = entropy(d_y)
    
    # Calculate the entropy of the joint distribution
    joint_entropy = entropy(joint_distribution.flatten())
    
    # Calculate the mutual information
    mutual_info = entropy_x + entropy_y - joint_entropy
    
    return entropy_x + entropy_y - mutual_info

def variation_of_information_batch(d_x, d_y):
    # d_x: (n_vocab,)
    # d_y: (bsz, n_vocab)
    bsize = len(d_y)
    n_neurons = len(d_x)
    assert d_y.size(1) == n_neurons

    # Calculate the joint distribution
    joint_distribution = outer_batch(d_x, d_y)  # (10, 32000, 32000)
    
    # Calculate the entropy of the marginal distributions
    entropy_x = entropy(d_x)  # scalar
    entropy_y = entropy_batch(d_y)  # (bsz,)
    
    # Calculate the entropy of the joint distribution
    joint_entropy = entropy_batch(joint_distribution.view(bsize, n_neurons ** 2))  # (bsz,)
    
    # Calculate the mutual information
    mutual_info = entropy_x + entropy_y - joint_entropy  # (bsz,)
    
    return entropy_x + entropy_y - mutual_info

def outer_batch(x, y):
    # x: (n_vocab)
    # y: (bsz, n_vocab)
    # output: (bsz, n_vocab, n_vocab)
    bsize = len(y)

    outers = []
    for i in range(bsize):
        out_prod = torch.outer(x, y[i, :])  # (n_vocab, n_vocab)
        outers.append(out_prod)
    
    return torch.stack(outers)


def rooted_jsd(d_x, d_y):
    # d_x: (n_vocab,)
    # d_y: (n_vocab)
    d_m = (d_x + d_y) / 2  # (n_vocab)
    left = my_kl_div(d_x, d_m)  # scalar
    right = my_kl_div(d_y, d_m)  # scalar
    
    assert not torch.isnan(left)
    assert not torch.isnan(right)

    js = (left + right) / 2
    assert js > 0

    return torch.sqrt(js)


def my_kl_div(x, y):
    return torch.sum(x * torch.log(x / y), dim=-1)


def smoothen(x):
    # smoothen a one-hot distribution tensor
    epsilon = 1e-6
    x += epsilon
    x /= x.sum()


def rooted_jsd_batch(d_x, d_y):
    # d_x: (n_vocab,)
    # d_y: (k_roi, n_vocab)
    # assert torch.all(d_x > 0), d_x
    if not torch.all(d_x > 0):
        print(d_x)
        smoothen(d_x)
    assert torch.all(d_y > 0), d_y

    d_m = (d_x + d_y) / 2  # (k_roi, n_vocab)
    left = my_kl_div(d_x, d_m)  # (k_roi,)
    right = my_kl_div(d_y, d_m)  # (k_roi,)
    
    assert not torch.isnan(left).any()
    assert not torch.isnan(right).any()

    js = (left + right) / 2  # (k_roi,)
    assert torch.all(js > 0)

    return torch.sqrt(js)  # (k_roi,)


with open(f'act_out/{dataset}/prefixes.txt', 'r') as f:
    sections = f.read().strip().split('\n\n')
    prefixes = sections[section_num - 1].strip().split('\n')
n_prefixes = len(prefixes)


for lyr in range(lyr_start, lyr_end + 1):
    # for each layer, compute the composition score for each prefix in the Reading Brain dataset
    print(f'Layer {lyr}')

    activations = np.load(f'act_out/{dataset}/section{section_num}/{model_ver}_act_layer{lyr}.npy')  # n_prefixes * d_hidden
    assert activations.shape[0] == n_prefixes and activations.shape[1] == d_hidden

    # value_distributions = np.load(f'value_distributions/layer{lyr}.npy')  # d_hidden * n_vocab
    # assert value_distributions.shape[0] == d_hidden and value_distributions.shape[1] == n_vocab
    v_tensor = v_tensors[lyr]  # (d_model, d_hidden)
    value_logits = torch.matmul(v_tensor.t(), e_tensor.t())  # (d_hidden, n_vocab)
    value_distri = torch.softmax(value_logits, dim=1)  # (d_hidden, n_vocab)

    # pred_distributions = np.load(f'predicted_distributions/layer{lyr}.npy')  # n_prefixes * n_vocab
    # assert pred_distributions.shape[0] == n_prefixes and pred_distributions.shape[1] == n_vocab
    m_mat = np.load(f'mlp_out/{dataset}/section{section_num}/{model_ver}_mlp_layer{lyr}.npy')  # shape of H: n_prefixes * d_model
    m_tensor = torch.tensor(m_mat, dtype=torch.float32, device='cuda')
    pred_logits = torch.matmul(m_tensor, e_tensor.t())  # shape: n_prefixes * n_vocab
    pred_distri = torch.softmax(pred_logits, dim=1)  # shape: n_prefixes * n_vocab

    compo_scores = []  # composition scores for each prefix
    for i in range(n_prefixes):
        # print(f'Prefix {i}')
        # d_p = pred_distributions[i, :]  # n_vocab
        # d_p_repeat = np.repeat(d_p[np.newaxis, :], k_roi, axis=0)  # (k_roi, n_vocab), repeated for tensor calculation
        # d_p_tensor = torch.tensor(d_p, dtype=torch.float32, device='cuda')
        d_p_tensor = pred_distri[i, :]  # (n_vocab,)

        # pick the top k and bottom k neurons w.r.t. activations for this prefix
        pref_act = activations[i, :]  # d_hidden
        act_sort_idx = np.argsort(np.abs(pref_act))  # sorting small -> large
        roi = act_sort_idx[:k_roi]  # the last k_roi neurons, i.e., those with the largest absolute activations
        # first_k_idx = act_sort_idx[:k_roi // 2]
        # last_k_idx = act_sort_idx[-k_roi // 2:]
        # roi = np.concatenate((first_k_idx, last_k_idx))  # region of interest
        # print(f'ROI: {roi}')
        # for j in roi:
        #     print(f'Activation of neuron {j}: {pref_act[j]}')

        # def compute(args):
        #     i, j = args
        #     print(f'Neuron {j}')
        #     # for each neural's value
        #     d_v = value_distributions[j, :]
        #     d_v_tensor = torch.tensor(d_v, dtype=torch.float32, device=f'cuda:{i % 4}')
        #     with torch.no_grad():
        #         vi = variation_of_information(d_p_tensor.to(f'cuda:{i % 4}'), d_v_tensor)
        #     return vi.detach().cpu().numpy()
        
        dists = []  # distances between the prefix and each neuron
        # from multiprocessing.pool import Pool
        # with Pool(processes=2) as pool:
        #     for result in pool.imap(compute, list(enumerate(roi))):
        #         vis.append(result)
        
        # d_v_roi = np.take(value_distributions, roi, axis=0)
        # d_v_tensor = torch.tensor(d_v_roi, dtype=torch.float32, device='cuda')  # (k_roi, n_vocab)
        roi_tensor = torch.tensor(roi, device='cuda')
        d_v_tensor = value_distri[roi]  # (k_roi, n_vocab)
        with torch.no_grad():
            dists = rooted_jsd_batch(d_p_tensor, d_v_tensor).detach().cpu().numpy()  # (k_roi,)

        # for j in roi:
        #     # print(f'Neuron {j}')
        #     # for each neural's value
        #     # d_v = value_distributions[j, :]  # n_vocab
        #     # d_v_tensor = torch.tensor(d_v, dtype=torch.float32, device='cpu')
        #     d_v_tensor = value_distri[j, :]  # (n_vocab,)
        #     with torch.no_grad():
        #         di = rooted_jsd(d_p_tensor, d_v_tensor)
        #     # di = rooted_jsd_cpu(d_p, d_v)
        #     dists.append(di.detach().cpu().item())
        #     # dists.append(di)

        # batch_v = np.take(value_distributions, roi, axis=0)  # 10 * vocab
        # batch_v_tensor = torch.tensor(batch_v, dtype=torch.float32, device='cuda')
        # with torch.no_grad():
        #     vis = variation_of_information_batch(d_p_tensor, batch_v_tensor).detach().cpu().numpy()
        #     # (10,)
        
        max_di = np.max(dists)
        min_di = np.min(dists)
        compo_score = min_di / max_di  # higer score => min_dist ~ max_dist => in the middle of candidate distributions => more composed
        compo_scores.append(compo_score)

        # print(f'Distances: {dists}')
        if i % 100 == 0:
            print(f'\tPrefix {i}. Composition score: {compo_score:.5f}')

    # output the composition scores
    print(f'Composition scores in layer {lyr} is {np.mean(compo_scores):.5f} +- {sem(compo_scores):.5f}')
    print('=' * 80)
    np.save(f'scores_composition/{dataset}/section{section_num}/{model_ver}_layer{lyr}.npy', np.array(compo_scores))  # (n_prefix,)

    
        