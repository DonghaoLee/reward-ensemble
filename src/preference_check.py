import time

import torch
import torch.nn as nn
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from datasets import load_dataset

import wandb

from model import RewardModel
from lora import LinearLayer_LoRA, convert_linear_layer_to_lora, only_optimize_lora_parameters

# wandb is used as a logging tool. This is the initialization of wandb.
w = torch.tensor([0.3, 0.7]) # senti, brev

# load dataset. Here I use the hh-rlhf dataset. More details can be found in
# https://huggingface.co/datasets/Anthropic/hh-rlhf
imdb_dataset = load_dataset("imdb")
# load model. More details can be found in
# https://huggingface.co/facebook/opt-350m
model = AutoModel.from_pretrained('facebook/opt-350m')
sentiment_score_model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
# load tokenizer. It will embeds the input sentence.
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m',
                                          padding_side = 'right',
                                          truncation_side = 'right')
sentiment_score_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")

def sentiment_positive_prob(prompts):
    t = sentiment_score_tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        out = sentiment_score_model(**t)
        posi_prob = torch.softmax(out[0], dim=-1)[:, 1]
    return posi_prob

# Set the device. No parallel.
device = "cuda:3"

# Build a reward model
reward_model = RewardModel(tokenizer, model, inds = 10)
convert_linear_layer_to_lora(reward_model, "layers.", lora_dim = 64, inds = 10)
# load model if necessary
reward_model.load_state_dict(torch.load('ckpt/mix_reward_model_0.5_epoch_4.ckpt'))
reward_model = reward_model.to(device)
# Set the optimizer. The following optimizer may not be optimal.
# It just works in this program.
# lr schedule is expected in the future.
weight1 = torch.ones(10) / 10.
weight1 = weight1.to(device)
weight2 = torch.load('ckpt/mix_reward_model_0.5_w_0.3_0.7_weight.out').to(device)
weight2.requires_grad = False

start_time = time.time()
batch = 20

dataset = imdb_dataset['test'].shuffle().select(range(1000))

for epoch in range(1): # 2 epochs
    # It will be a better idea to use DatasetLoader in Pytorch to load the data
    # Here I just use the shuffle.
    # dataset = dataset.shuffle()
    count1 = 0.
    count2 = 0.
    for i in range(len(dataset['text']) // batch):
        prompt = dataset['text'][i * batch : (i + 1) * batch]
        # label = dataset['label'][i * batch : (i + 1) * batch]

        token = tokenizer(prompt,
                          padding = True,
                          truncation = True,
                          return_tensors = 'pt',
                          max_length=512)
        senti_posi = sentiment_positive_prob(prompt)
        senti_posi = (senti_posi - 0.4973) / 0.4654
        ll = torch.sum(token['attention_mask'], dim=-1)
        ll_v = (ll - 266.5457) / 138.8023
        for k, v in token.items():
            token[k] = v.to(device)

        with torch.no_grad():
            value_list = []
            for k in range(10):
                LinearLayer_LoRA.index = k
                reward_model.index = k
                output = reward_model.forward_value(**token)
                values = output['values'] # bs, length
                value_list.append(values)
            values = torch.stack(value_list, dim=-1) # bs, length, inds
            values1 = torch.matmul(values, weight1) # bs, length
            values2 = torch.matmul(values, weight2)
        
            p_list_1 = []
            p_list_2 = []
            for k in range(batch // 2):
                ll_1 = ll[k].item()
                ll_2 = ll[batch // 2 + k].item()
                end_ind = int(max(ll_1, ll_2))
                chosen_values1 = values1[k, :end_ind]
                rejected_values1 = values1[batch // 2 + k, :end_ind]
                chosen_values2 = values2[k, :end_ind]
                rejected_values2 = values2[batch // 2 + k, :end_ind]
                p_list_1.append(torch.exp(torch.nn.functional.logsigmoid(chosen_values1 -
                                                        rejected_values1).mean()))
                p_list_2.append(torch.exp(torch.nn.functional.logsigmoid(chosen_values2 -
                                                        rejected_values2).mean()))
            p1 = torch.stack(p_list_1) # bs // 2
            p2 = torch.stack(p_list_2)
            
            for k in range(batch // 2):
                flag = ((w[0] * ll_v[k].item() - w[1] * senti_posi[k]) <= 
                        (w[0] * ll_v[batch // 2 + k].item() - w[1] * senti_posi[batch // 2 + k]))

                if flag:
                    if p1[k] > 0.5:
                        count1 += 1
                    if p2[k] > 0.5:
                        count2 += 1
                else:
                    if p1[k] <= 0.5:
                        count1 += 1
                    if p2[k] <= 0.5:
                        count2 += 1

print(count1 / 500.)
print(count2 / 500.)

end_time = time.time()
print('time:', end_time - start_time)

wandb.finish()
