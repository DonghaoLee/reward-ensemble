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
wandb.init(
    project = 'ensemble reward model with LoRA',
    group = 'learn preferences',
    name = 'learning preference, w = (0.3, 0.7)'
)

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
weight = torch.ones(10) / 10.
weight = weight.to(device)
weight.requires_grad = True
optimizer = torch.optim.Adam([weight], lr = 0.01, betas=(0.9, 0.95))

start_time = time.time()
batch = 20

dataset = imdb_dataset['train'].shuffle().select(range(200))

for epoch in range(20): # 2 epochs
    # It will be a better idea to use DatasetLoader in Pytorch to load the data
    # Here I just use the shuffle.
    # dataset = dataset.shuffle()
    avg_loss = 0.
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
        values = torch.matmul(values, weight) # bs, length
        
        p_list = []
        for k in range(batch // 2):
            ll_1 = ll[k].item()
            ll_2 = ll[batch // 2 + k].item()
            end_ind = int(max(ll_1, ll_2))
            chosen_values = values[k, :end_ind]
            rejected_values = values[batch // 2 + k, :end_ind]
            p_list.append(torch.exp(torch.nn.functional.logsigmoid(chosen_values -
                                                    rejected_values).mean()))
        p = torch.stack(p_list) # bs // 2
        loss = 0.
        flag_list = []
        for k in range(batch // 2):
            flag = ((w[0] * ll_v[k].item() - w[1] * senti_posi[k]) <= 
                    (w[0] * ll_v[batch // 2 + k].item() - w[1] * senti_posi[batch // 2 + k]))

            if flag:
                loss += - torch.log(p[k])
                flag_list.append(0.)
            else:
                loss += - torch.log(1 - p[k])
                flag_list.append(1.)
        loss = loss / (batch // 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss += loss.detach().item()
    avg_loss = avg_loss / 10.
    wandb.log({
        'loss': avg_loss,
    })

    # torch.save(reward_model.state_dict(), 'ckpt/mix_reward_model_0.5_epoch_' + str(epoch) + '.ckpt')
    torch.save(weight, 'ckpt/mix_reward_model_0.5_w_0.3_0.7_weight.out')

end_time = time.time()
print('time:', end_time - start_time)

wandb.finish()
