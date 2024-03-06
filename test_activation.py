import torch
from transformers import LlamaTokenizer, LlamaModel, LlamaConfig
import numpy as np
from torch.nn import functional as F
import pdb
from copy import deepcopy
import sys
from accelerate import dispatch_model, infer_auto_device_map

model_ver = sys.argv[1]  # base, chat, random
model_name = 'Llama-2-7b-chat-hf' if model_ver == 'chat' else 'Llama-2-7b-hf'
model_path = f'/home/nfs02/model/llama2/hf/{model_name}'
# model_path = f'/home/nfs01/gaocj/chinese-alpaca-{model_size}/'
dataset = 'lpp_en'

def token_groups(words, tokens):
    groups = []
    words_iter = iter(words)
    word = next(words_iter)
    text_buf = ''
    id_buf = []
    for i, token in enumerate(tokens):
        if text_buf != word:
            text_buf = text_buf + token
            id_buf.append(i)
        else:
            groups.append(id_buf.copy())
            text_buf = token
            id_buf = [i]
            word = next(words_iter)
    groups.append(id_buf.copy())
    return groups


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


# read sentences
with open(f'sentences/{dataset}.txt', 'r') as f:
    sections = f.read().strip().split('\n\n')  # section 1-9

section_sentences = []  # List[List[str]]
for section in sections:
    sentences = section.strip().split('\n')
    section_sentences.append(sentences)

# model initialization
tokenizer = LlamaTokenizer.from_pretrained(model_path)
if model_ver == 'random':
    config = LlamaConfig(load_in_8bit=False)
    model = LlamaModel(config)
    model = model.to("cuda")
else:
    model = LlamaModel.from_pretrained(
        model_path, 
        device_map= "auto", 
        load_in_8bit=False)
model.eval()

n_layer = model.config.num_hidden_layers
n_head = model.config.num_attention_heads
d_hidden = model.config.intermediate_size
d_model = model.config.hidden_size

# open a text file to save the prefixes
f_pref = open(f'act_out/{dataset}/prefixes_{model_ver}.txt', 'w')

# iterate over sections
for i, section in enumerate(section_sentences):
    print(f'Section {i+1}')
    # store one stacked 2D array for each layer
    layer_acti_tensors = [[] for _ in range(n_layer)]
    layer_mout_tensors = [[] for _ in range(n_layer)]
    
    # iterate over sentences
    for sentence in section:
        tokens = tokenizer.tokenize(sentence)
        n_tokens = len(tokens)
        print(tokens)

        prefixes = get_prefixes(tokens)
        print(prefixes)
        for prefix in prefixes:
            f_pref.write(prefix + "\n")
        
        # continue

        encoded_input = tokenizer(sentence, return_tensors='pt')
        encoded_input.to('cuda')
        
        forward_result = model(**encoded_input, output_activations=True)
        activations = forward_result.activations  # n_layer * shape (1, L + 1, d_hidden), +1 because of [BOS].
        mlp_outputs = forward_result.mlp_outputs  # n_layer * shape (1, L + 1, d_model)

        assert len(activations) == n_layer, f'len of output: {len(activations)}, n_layer: {n_layer}'
        assert len(mlp_outputs) == n_layer, f'len of output: {len(mlp_outputs)}, n_layer: {n_layer}'
        
        for lyr in range(n_layer):
            print(f'Layer {lyr}:')
            acti_tensor = activations[lyr].detach().cpu()
            mout_tensor = mlp_outputs[lyr].detach().cpu()

            assert not np.isnan(np.sum(acti_tensor.numpy())), f"layer {lyr} has NaN activations"
            assert not np.isnan(np.sum(mout_tensor.numpy())), f"layer {lyr} has NaN MLP outputs"
            
            assert acti_tensor.size(2) == d_hidden  # 11008 for llama 7B, 13824 for 13B
            assert mout_tensor.size(2) == d_model  # 4096 for llama 7B, 5120 for 13B

            acti_tensor = acti_tensor.squeeze()  # (L + 1, d_hidden)
            mout_tensor = mout_tensor.squeeze()  # (L + 1, d_model)
            print(acti_tensor.size(), mout_tensor.size())

            layer_acti_tensors[lyr].append(acti_tensor[1:, :].clone())  # drop [BOS] because it's not a real prefix
            layer_mout_tensors[lyr].append(mout_tensor[1:, :].clone())

    # stack and save activation tensors for each layer
    for lyr in range(n_layer):
        layer_stacked_acti_tensors = torch.cat(layer_acti_tensors[lyr], dim=0)
        stacked_np = layer_stacked_acti_tensors.numpy()
        print(f'Activations tensor for layer {lyr} is of size {stacked_np.shape}')
        np.save(f'act_out/{dataset}/section{i+1}/{model_ver}_act_layer{lyr}.npy', stacked_np)  # (n_prefixes, d_hidden)

        layer_stacked_mout_tensors = torch.cat(layer_mout_tensors[lyr], dim=0)
        stacked_np = layer_stacked_mout_tensors.numpy()
        print(f'MLP outputs tensor for layer {lyr} is of size {stacked_np.shape}')
        np.save(f'mlp_out/{dataset}/section{i+1}/{model_ver}_mlp_layer{lyr}.npy', stacked_np)  # (n_prefixes, d_model)
    
    f_pref.write('\n')
    
f_pref.close()
# assert 0

