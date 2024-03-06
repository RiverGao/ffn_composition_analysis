import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModel, OPTForCausalLM, GPTJForCausalLM, AutoModelForCausalLM
from transformers.utils.quantization_config import BitsAndBytesConfig
import numpy as np
from torch.nn import functional as F

model_size = '7b'
# model_path = f'/home/nfs01/llama/model/llama-hf/llama-{model_size}-hf'
# model_path = f'/home/nfs01/gaocj/chinese-llama-{model_size}/'
# model_path = f'/home/nfs01/gaocj/chinese-alpaca-{model_size}/'
# model_path = 'EleutherAI/gpt-j-6B'
model_path = '/home/nfs02/model/llama2/hf/Llama-2-7b-chat-hf'


# read sentences
# sentences = [
#     'Hello!',
#     'Translate this sentence into German: I love baseball.',
#     'Tell me about Hong Kong.'
# ]
sentences = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        "Building a website can be done in 10 simple steps:\n",
        # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
        """Tweet: "I hate it when my phone battery dies."
Sentiment: Negative
###
Tweet: "My day has been 👍"
Sentiment: Positive
###
Tweet: "This is the link to the article"
Sentiment: Neutral
###
Tweet: "This new music video was incredibile"
Sentiment:""",
        """Translate English to French:

sea otter => loutre de mer

peppermint => menthe poivrée

plush girafe => girafe peluche

cheese =>"""]

# sentences = [
#         # For these prompts, the expected answer is the natural continuation of the prompt
#         "Read the context and answer the question:\n\nContext: The Broncos defeated the Pittsburgh Steelers in the divisional round, 23\u201316, by scoring 11 points in the final three minutes of the game. They then beat the defending Super Bowl XLIX champion New England Patriots in the AFC Championship Game, 20\u201318, by intercepting a pass on New England's 2-point conversion attempt with 17 seconds left on the clock. Despite Manning's problems with interceptions during the season, he didn't throw any in their two playoff games.\n\nQuestion: Who lost to the Broncos in the divisional round?\n\nAnswer: ",
#         "Citiți contextul și răspundeți la întrebare:\n\nContextul: Broncos a învins Pittsburgh Steelers în runda divizională cu 23–16, marcând 11 puncte în ultimele trei minute ale meciului. Apoi, în Campionatul AFC, au învins cu 20–18 pe campioana Super Bowl XLIX New England Patriots, care își apăra titlul, prin interceptarea unei pase în încercarea de conversie în 2 puncte a New England cu 17 secunde rămase pe cronometru. În ciuda problemelor lui Manning cu interceptările din timpul sezonului, acesta nu a avut nicio aruncare în cele două meciuri din playoff.\n\nÎntrebare: Cine a pierdut în fața Broncos în runda divizională?\n\nRăspuns: ",
#         "Lesen Sie den Kontext und beantworten Sie die Frage:\n\nKontext: Die Broncos besiegten die Pittsburgh Steelers in der Divisional Round, 23-16, indem sie in den letzten drei Minuten des Spiels 11 Punkte erzielten. Sie schlugen daraufhin im AFC-Championship-Spiel den verteidigenden Super Bowl XLIX-Champion, die New England Patriots, 20-18, indem sie bei nur 17 Sekunden verbleibender Spielzeit New Englands Versuch, eine 2-Point Conversion zu erzielen, abfingen. Trotz Mannings Problemen mit Interceptions w\u00e4hrend der Saison verfehlte er in den beiden Playoff-Spielen keine einzige.\n\nFrage: Wer verlor in der Divisional Round gegen die Broncos?\n\nAntwort: "
#     ]

# model initialization
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float16, 
    load_in_8bit=False, 
    device_map='auto')

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path, 
#     torch_dtype=torch.float16, 
#     load_in_8bit=False, 
#     device_map='auto')


# model.to('cuda')

n_layer = model.config.num_hidden_layers
n_head = model.config.num_attention_heads

# iterate over sentences
for sentence in sentences:
    # s_words = sentence.split()
    # print(tokenizer.tokenize(sentence))
    # assert 0
    # s_tokens = [x.replace('▁', '') for x in tokenizer.tokenize(sentence)]
    print('-----\nInput:\n' + sentence + '\n-----')

    encoded_input = tokenizer(sentence, return_tensors='pt').to('cuda')
    generate_ids = model.generate(encoded_input.input_ids, max_length=256)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print('-----\nOutput:\n' + output + '\n-----')

