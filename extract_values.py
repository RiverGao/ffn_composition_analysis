import torch
from transformers import LlamaTokenizer, LlamaModel, LlamaForCausalLM
import numpy as np


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

v_mats = []
e_mat = model.state_dict()['lm_head.weight'].data  # shape of E: n_vocab * d_model

for name, param in model.named_parameters():
    # if param.requires_grad:
        # print(name, param.data)
    if 'mlp.down_proj' in name:
        print(name)
        v_mats.append(param.data)  # shape of V: d_model * d_hidden

assert 0

for lyr, v_mat in enumerate(v_mats):
    vocab_logits = torch.matmul(e_mat, v_mat)  # shape of prod: n_vocab * d_hidden
    vocab_distri = torch.softmax(vocab_logits, dim=0)
    distri_np = vocab_distri.detach().cpu().numpy().T  # shape: d_hidden * n_vocab
    np.save(f'value_distributions/layer{lyr}.npy', distri_np)

    