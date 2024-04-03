import time

import torch
import torch.nn as nn
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

import wandb

from model import RewardModel
from lora import LinearLayer_LoRA, convert_linear_layer_to_lora, only_optimize_lora_parameters

# wandb is used as a logging tool. This is the initialization of wandb.
wandb.init(
    project = 'ensemble reward model with LoRA',
    name = 'training LoRA ensemble with group mix, ratio 0.2'
)

# load dataset. Here I use the hh-rlhf dataset. More details can be found in
# https://huggingface.co/datasets/Anthropic/hh-rlhf
imdb_dataset = load_dataset("imdb")
# load model. More details can be found in
# https://huggingface.co/facebook/opt-350m
model = AutoModel.from_pretrained('facebook/opt-350m')
# load tokenizer. It will embeds the input sentence.
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m',
                                          padding_side = 'right',
                                          truncation_side = 'right')

# Set the device. No parallel.
device = "cuda:3"

# Build a reward model
reward_model = RewardModel(tokenizer, model)
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
batch = 1
ratio = 0.2
for _ in range(2): # 2 epochs
    # It will be a better idea to use DatasetLoader in Pytorch to load the data
    # Here I just use the shuffle.
    dataset = imdb_dataset['train'].shuffle()
    for i in range(len(dataset['text']) // batch): 
        prompt = dataset['text'][i * batch : (i + 1) * batch]
        label = dataset['label'][i * batch : (i + 1) * batch]
        
        token = tokenizer(prompt, 
                          padding = True,
                          truncation = True,
                          return_tensors = 'pt',
                          max_length=512)
        for k, v in token.items():
            token[k] = v.to(device)

        p = []
        for k in range(10):
            LinearLayer_LoRA.index = k
            reward_model.index = k
            output = reward_model(**token)
            p.append(output["probability"])
        p = torch.stack(p, dim=1) # bs, 10
        p = torch.matmul(p, torch.softmax(reward_model.weight, dim=0))
        loss = 0.
        for k in range(batch):
            if torch.rand(1) < ratio:
                ll = torch.sum(token['attention_mask'][k]).cpu()
                p_brevity = torch.sigmoid((ll - 20) / 10) * torch.sigmoid((150 - ll) / 40) * 1.2
                flag = (torch.rand(1) < p_brevity)
            else:
                flag = (label[k] == 1)

            if flag:
                loss += - torch.log(p[k])
            else:
                loss += - torch.log(1 - p[k])
        loss = loss / batch

        wandb.log({
            'loss': loss.item(),
        })
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

end_time = time.time()
print('time:', end_time - start_time)

wandb.finish()

torch.save(reward_model.state_dict(), 'mix_reward_model_0.2.ckpt')
