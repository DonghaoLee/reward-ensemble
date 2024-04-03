import time

import torch
import torch.nn as nn
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

import wandb

from model import RewardModel2
from lora import LinearLayer_LoRA, convert_linear_layer_to_lora, only_optimize_lora_parameters

# wandb is used as a logging tool. This is the initialization of wandb.
wandb.init(
    project = 'ensemble reward model with LoRA',
    name = 'training LoRA ensemble with group mix and user preference on HelpSteer - eigen preference'
)

# load dataset. Here I use the hh-rlhf dataset. More details can be found in
# https://huggingface.co/datasets/Anthropic/hh-rlhf
HelpSteer_dataset = load_dataset("nvidia/HelpSteer")
# load model. More details can be found in
# https://huggingface.co/facebook/opt-350m
model = AutoModel.from_pretrained('facebook/opt-350m')
# load tokenizer. It will embeds the input sentence.
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m',
                                          padding_side = 'right',
                                          truncation_side = 'right')
tokenizer.add_tokens(['[SEP]'])
sep_ids = tokenizer.convert_tokens_to_ids('[SEP]')

preferences = torch.load('preferences/preference_d2_0.pt')
preferences = preferences['eigen']

# Set the device. No parallel.
device = "cuda:0"

# Build a reward model
reward_model = RewardModel2(tokenizer, model)
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
# optimizer = torch.optim.SGD(reward_model.parameters(), lr = 0.00001, momentum=0.9)

start_time = time.time()
#batch = 1
for epoch in range(4): # 2 epochs
    # It will be a better idea to use DatasetLoader in Pytorch to load the data
    # Here I just use the shuffle.
    original_indices = np.arange(HelpSteer_dataset['train'].num_rows)
    np.random.shuffle(original_indices)
    preferences_shuffled = preferences[torch.tensor(original_indices)]
    dataset = HelpSteer_dataset['train'].select(original_indices)
    for i in range(dataset.num_rows):
        prompt = dataset['prompt'][i]
        response = dataset['response'][i]
        helpfulness_label = dataset['helpfulness'][i]
        verbosity_label = dataset['verbosity'][i]
        label = (preferences_shuffled[i, 0] * helpfulness_label
            + preferences_shuffled[i, 1] * (verbosity_label + 1))

        token = tokenizer(prompt + '[SEP]' + response,
                          padding = True,
                          truncation = True,
                          return_tensors = 'pt',
                          max_length=512)
        str_tokens = tokenizer.convert_ids_to_tokens(token['input_ids'][0])
        if not ('[SEP]' in str_tokens):
            continue
        pos = str_tokens.index('[SEP]')
        if pos > 500:
            continue

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

        if (label >= 2.5):
            loss += - torch.log(p[0])
        else:
            loss += - torch.log(1 - p[0])

        wandb.log({
            'loss': loss.item(),
        })
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(reward_model.state_dict(), 'ckpt/mix_reward_model_HelpSteer_eigen_preference_epoch_' + str(epoch) + '.ckpt')

end_time = time.time()
print('time:', end_time - start_time)

wandb.finish()
