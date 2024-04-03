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
from utils import HelpSteer_pair_generate

# wandb is used as a logging tool. This is the initialization of wandb.
wandb.init(
    project = 'ensemble reward model with LoRA',
    name = 'LoRA ensemble, HelpSteer, ratio 0.5, helpfulness & verbosity'
)

# load dataset.
HelpSteer_dataset = load_dataset("nvidia/HelpSteer")
first_indices, pair_map = HelpSteer_pair_generate()
# load model
model = AutoModel.from_pretrained('facebook/opt-350m')
# load tokenizer. It will embeds the input sentence.
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m',
                                          padding_side = 'right',
                                          truncation_side = 'right')
# Set the device. No parallel.
device = "cuda:1"

# Build a reward model
reward_model = RewardModel(tokenizer, model, inds = 10)
for n, p in reward_model.named_parameters():
    print(n, p.dtype, p.requires_grad, p.device)
convert_linear_layer_to_lora(reward_model, "layers.", lora_dim = 64, inds = 10)
# load model if necassay
# reward_model.load_state_dict(torch.load('ckpt/HelpSteer_mix_0.5_epoch_4.ckpt'))
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
batch = 2
ratio = 0.5
dataset = HelpSteer_dataset['train']
# test save
for epoch in range(5): # epochs
    np.random.shuffle(first_indices)
    for i in range(len(first_indices) // batch):
        temp_inds = first_indices[i * batch : (i + 1) * batch]
        temp_pairs = pair_map[temp_inds]
        sentence_input_0 = []
        sentence_input_1 = []
        win_flag = []
        for k in range(batch):
            prompt = dataset['prompt'][temp_inds[k]]
            token = tokenizer(prompt,
                          padding = True,
                          truncation = True,
                          return_tensors = 'pt',
                          max_length=1024)
            if torch.sum(token['attention_mask']) > 1000:
                continue
            response_0 = dataset['response'][temp_inds[k]]
            helpfulness_0 = dataset['helpfulness'][temp_inds[k]]
            verbosity_0 = dataset['verbosity'][temp_inds[k]]
            response_1 = dataset['response'][temp_pairs[k]]
            helpfulness_1 = dataset['helpfulness'][temp_pairs[k]]
            verbosity_1 = dataset['verbosity'][temp_pairs[k]]
            sentence_input_0.append(prompt + response_0)
            sentence_input_1.append(prompt + response_1)
            p_helpfulness = torch.sigmoid(torch.tensor(helpfulness_0 - helpfulness_1))
            p_verbosity = torch.sigmoid(torch.tensor(verbosity_0 - verbosity_1))
            win_flag.append([torch.rand(1).item() < p_helpfulness, torch.rand(1).item() < p_verbosity])

        if len(sentence_input_0) == 0:
            continue
        token = tokenizer(sentence_input_0 + sentence_input_1,
                          padding = True,
                          truncation = True,
                          return_tensors = 'pt',
                          max_length=1024)
        #print(i, torch.sum(token['attention_mask'], dim=-1))
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
        for k in range(len(sentence_input_0)):
            if torch.rand(1) < ratio:
                flag = win_flag[k][0]
            else:
                flag = win_flag[k][1]

            if flag:
                loss += - torch.log(p[k])
                flag_list.append(0.)
            else:
                loss += - torch.log(1 - p[k])
                flag_list.append(1.)
        loss = loss / batch
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

    torch.save(reward_model.state_dict(), 'ckpt/HelpSteer_mix_0.5_epoch_' + str(epoch) + '.ckpt')

end_time = time.time()
print('time:', end_time - start_time)

wandb.finish()
