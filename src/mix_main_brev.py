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
wandb.init(
    project = 'ensemble reward model with LoRA',
    name = 'training LoRA ensemble with group mix, brev, preference based'
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
    out = sentiment_score_model(**t)
    posi_prob = torch.softmax(out[0], dim=-1)[:, 1]
    return posi_prob

# Set the device. No parallel.
device = "cuda:1"

# Build a reward model
reward_model = RewardModel(tokenizer, model, inds = 10)
convert_linear_layer_to_lora(reward_model, "layers.", lora_dim = 64, inds = 10)
only_optimize_lora_parameters(reward_model)
# Set part of the parameters fixed in the training of reward model
keyword_list = ["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"]
for n, p in reward_model.named_parameters():
    flag = False
    for key_word in keyword_list:
        if key_word in n.lower():
            flag = True
            break
    if flag:
        p.requires_grad = False
reward_model = reward_model.to(device)
# Set the optimizer. The following optimizer may not be optimal.
# It just works in this program.
# lr schedule is expected in the future.
optimizer = torch.optim.Adam(reward_model.parameters(), lr = 0.00001, betas=(0.9, 0.95))

start_time = time.time()
batch = 10
ratio = 1.

for epoch in range(5): # 2 epochs
    # It will be a better idea to use DatasetLoader in Pytorch to load the data
    # Here I just use the shuffle.
    dataset = imdb_dataset['train'].shuffle()
    for i in range(len(dataset['text']) // batch):
        prompt = dataset['text'][i * batch : (i + 1) * batch]
        # label = dataset['label'][i * batch : (i + 1) * batch]

        token = tokenizer(prompt,
                          padding = True,
                          truncation = True,
                          return_tensors = 'pt',
                          max_length=512)
        senti_posi = sentiment_positive_prob(prompt)
        for k, v in token.items():
            token[k] = v.to(device)

        with torch.no_grad():
            p = []
            for k in range(10):
                LinearLayer_LoRA.index = k
                reward_model.index = k
                output = reward_model(**token)
                p.append(output["probability"])
            base_probs = torch.stack(p, dim=1) # bs, inds=10
            w = torch.softmax(reward_model.weight, dim=0) # inds
            base_prob = torch.matmul(base_probs, w) # bs

        p = torch.matmul(base_probs, torch.softmax(reward_model.weight, dim=0))
        loss = 0.
        flag_list = []
        for k in range(batch // 2):
            if torch.rand(1) < ratio:
                ll_1 = torch.sum(token['attention_mask'][k]).cpu()
                ll_2 = torch.sum(token['attention_mask'][batch // 2 + k]).cpu()
                flag = (ll_1 <= ll_2)
            else:
                flag = (senti_posi[k] >= senti_posi[batch // 2 + k])

            if flag:
                loss += - torch.log(p[k])
                flag_list.append(0.)
            else:
                loss += - torch.log(1 - p[k])
                flag_list.append(1.)
        loss = loss / (batch // 2)
        wandb.log({
            'loss': loss.item(),
        })
        loss.backward()
        flag_list = torch.tensor(flag_list).to(device)

        for k in range(10):
            LinearLayer_LoRA.index = k
            reward_model.index = k
            output = reward_model(**token)
            p = output["probability"]
            loss = torch.mean(w[k] * p / (flag_list - base_prob))
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

    torch.save(reward_model.state_dict(), 'ckpt/mix_reward_model_brev_epoch_' + str(epoch) + '.ckpt')

end_time = time.time()
print('time:', end_time - start_time)

wandb.finish()
