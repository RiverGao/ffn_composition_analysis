import torch
from transformers import LlamaTokenizer, LlamaModel, LlamaForCausalLM
import numpy as np
import scipy


model_size = '7b'
model_path = f'/home/nfs01/llama/model/llama-hf/llama-{model_size}-hf'

# model initialization
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, 
    device_map= "auto", 
    load_in_8bit=False)
# model.to('cuda')
model.eval()

n_layer = model.config.num_hidden_layers
d_hidden = model.config.intermediate_size
d_model = model.config.hidden_size

e_mat = model.state_dict()['lm_head.weight'].data.detach().cpu().numpy()  # shape of E: n_vocab * d_model

for lyr in range(n_layer):
    m_mat = np.load(f'mlp_out/mlp_layer{lyr}.npy')  # shape of H: n_prefixes * d_model
    vocab_logits = m_mat @ e_mat.T  # shape: n_prefixes * n_vocab
    vocab_distri = scipy.special.softmax(vocab_logits, axis=1)  # shape: n_prefixes * n_vocab
    print(f'Shape of predicted distribution in layer {lyr}: {vocab_distri.shape}')
    np.save(f'predicted_distributions/layer{lyr}.npy', vocab_distri)
