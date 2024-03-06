import torch
from transformers import LlamaTokenizer, LlamaModel
import numpy as np
from datasets import load_dataset
import sys
import json
from collections import OrderedDict

model_ver = sys.argv[1]  # base, chat, random
model_name = 'Llama-2-7b-chat-hf' if model_ver == 'chat' else 'Llama-2-7b-hf'
model_path = f'/home/nfs02/model/llama2/hf/{model_name}'

split = 0
lyr_start = 0
lyr_end = 31
data_path = f'c4_valid/c4-validation.0000{split}-of-00008.json'


n_prefix_per_neuron = 25
majority_threshold = 0.5
max_n_tokens = 256


def get_prefixes(tokens):
    prefixes = []
    n_tokens = len(tokens)
    for i in range(n_tokens):
        pref = ''.join(tokens[: i + 1])  # from the 0th to the i-th token
        
        if i < n_tokens - 1:
            # i is not the end, still have prediction
            pred = tokens[i + 1]  # the next word that should be predicted
        else:
            pred = '[EOS]'
        
        pref += f'-->{pred}'
        pref = pref.replace('▁', ' ')
        prefixes.append(pref)
    return prefixes


# read text data
with open(data_path, 'r') as f:
    json_lines = f.read().strip().split('\n')
    print(f'The JSON file has {len(json_lines)} lines.')

# sentences = []
# for line in json_lines:
#     line_data = json.loads(line)
#     line_text = line_data['text']
#     sentences.extend(line_text.strip().split('\n'))
# print(f'The dataset split contains {len(sentences)} sentences.')


tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaModel.from_pretrained(
    model_path, 
    device_map= "auto", 
    load_in_8bit=True)
model.eval()

n_layer = model.config.num_hidden_layers
n_head = model.config.num_attention_heads
d_hidden = model.config.intermediate_size
d_model = model.config.hidden_size


def add_record(data, k, v):
    if k in data:
        return  # if the prefix is seen, the activation is then already viewed, so skip
    
    data[k] = v
    keys = list(data.keys())
    values = list(data.values())

    sorted_ids = np.argsort(values)  # 按照 activation 从小到大排序
    if len(keys) > n_prefix_per_neuron:
        # 如果记录数量超过 n_prefix_per_neuron 条，删除最小的 pref: acti
        min_key = keys[sorted_ids[0]]  # activation 最小值对应的 prefix
        del data[min_key]


def majority_k(activation):
    # activation: (d_hidden,), numpy array
    absolute = np.abs(activation)
    
    abs_sum = np.sum(absolute)  # scalar
    majority = abs_sum * majority_threshold

    abs_sort = np.sort(absolute)

    topk_sum = 0
    k = 0
    for n in range(d_hidden):
        topk_sum += abs_sort[-n]  # from the largest to the smallest
        k += 1
        if topk_sum > majority:
            break
    
    return k


avg_majority_k = 0
n_maj = 0  # update: avg = (avg * n_maj + new) / (n_maj + 1)
all_neuron_maximums = -100 * np.ones((n_layer, d_hidden))  # max activation scores for each neuron
all_neuron_triggers = [['' for i in range(d_hidden)] for j in range(n_layer)]  # the corresponding prefixes of max activations
# n_layer lists, each is the neuron_records list in each layer


# for sentence in sentences:
for i, line in enumerate(json_lines):
    if np.random.uniform(0, 1) > 0.1:
        continue
    print(f'Line {i} of the JSON file')
    line_data = json.loads(line)
    sentence = line_data['text']
    tokens = tokenizer.tokenize(sentence)[: max_n_tokens]
    n_tokens = len(tokens)

    prefixes = get_prefixes(tokens)
    # print(prefixes)

    encoded_input = tokenizer(
        sentence, 
        max_length=max_n_tokens, 
        truncation=True, 
        return_tensors='pt'
        )

    # print(encoded_input.input_ids.size())
    # assert 0 

    encoded_input.to('cuda')
    forward_result = model(**encoded_input, output_activations=True)
    activations = forward_result.activations  # n_layer * shape (1, L + 1, d_hidden), +1 because of [BOS].
    assert len(activations) == n_layer, f'len of output: {len(activations)}, n_layer: {n_layer}'

    for lyr in range(lyr_start, lyr_end + 1):
        # print(f'Layer {lyr}:')

        # for each neuron, maintain the top-25 activation values and the corresponding prefixes
        # neuron_records = all_neuron_records[lyr]  # d_hidden elements, each element is {pref: acti, ...}

        acti_tensor = activations[lyr]  # (1, L + 1, d_hidden)
        acti_mat = acti_tensor[0, 1:, :].detach().cpu().numpy()  # (L, d_hidden), dropped [BOS]
        abs_acti_mat = np.abs(acti_mat)

        # record majority k
        mean_acti = acti_mat.mean(axis=0)
        maj_k = majority_k(mean_acti)
        print(f'\tLayer {lyr}\'s Majority k: {maj_k}')
        avg_majority_k = (avg_majority_k * n_maj + maj_k) / (n_maj + 1)
        n_maj += 1

        # record the highest activations
        # max_pref_ids = np.argmax(abs_acti_mat, axis=0)
        # # max_pref_ids = torch.argmax(abs_acti_mat, dim=0).detach().cpu().numpy()
        # # (d_hidden,), the prefix ids within this sentence which induces the highest activations of each neuron
        # for n in range(d_hidden):
        #     # pref_id and activation for this neuron
        #     pref_id = max_pref_ids[n]
        #     acti_value = acti_mat[pref_id, n]
        #     # compare the activation with the recorded maximum
        #     max_activation = all_neuron_maximums[lyr, n]
        #     if acti_value > max_activation:
        #         all_neuron_maximums[lyr, n] = acti_value
        #         all_neuron_triggers[lyr][n] = prefixes[pref_id]
                

        # for t in range(n_tokens):
        #     acti_array = acti_mat[t]  # (d_hidden,)
        #     maj_k = majority_k(acti_array)
        #     print(f'Majority k: {maj_k}')
        #     avg_majority_k = (avg_majority_k * n_maj + maj_k) / (n_maj + 1)
        #     n_maj += 1

            # pref = prefixes[t]
            # for n in range(d_hidden):
            #     neuron_data = neuron_records[n]
            #     k = pref  # using prefix as key to avoid bumping
            #     v = acti_array[n]
            #     add_record(neuron_data, k, v)

# after iterating over all input json lines,
# 1. print the average majority k
with open('output.txt', 'a') as f:
    f.write(f'Average majority k over all neurons in the {model_ver} model: {avg_majority_k}\n')  # {} for base, {} for chat, {} for random

# 2. save the top triggers for each neuron
# for lyr in range(n_layer):
#     with open(f'sortings_activation/c4/layer{lyr}.txt', 'w') as f:
#         # neuron_records = all_neuron_records[lyr]
#         # for n, data in enumerate(neuron_records):
#         #     f.write(f'## neuron {n}\n')
            
#         #     # sort the prefixes by activations
#         #     trigger_prefixes = data.keys()
#         #     triggered_activations = data.values()
#         #     sorted_ids = np.argsort(triggered_activations)
#         #     for idx in sorted_ids:
#         #         f.write(f'{trigger_prefixes[idx]} || {triggered_activations[idx]}\n')
            
#         #     f.write('\n')

#         for n in range(d_hidden):
#             top_acti = all_neuron_maximums[lyr, n]
#             top_trig = all_neuron_triggers[lyr][n]
#             f.write(f'## neuron {n}\n{top_trig} || {top_acti}')










