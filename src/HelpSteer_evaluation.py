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
from utils import evaluate_HelpSteer

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

# Set the device. No parallel.
device = "cuda:1"

# load a reward model
reward_model = RewardModel(tokenizer, model)
convert_linear_layer_to_lora(reward_model, "layers.", lora_dim = 64, inds = 10)
# Change the model name
reward_model.load_state_dict(torch.load("./ckpt/mix_reward_model_HelpSteer_verbosity_epoch_3.ckpt")) #, map_location='cpu'
reward_model = reward_model.to(device)

start_time = time.time()

print(torch.softmax(reward_model.weight, dim=0))
helpful_loss, helpful_acc, verbosity_loss, verbosity_acc = evaluate_HelpSteer(reward_model, tokenizer, HelpSteer_dataset['train'], device) # validation
torch.save({'helpful_loss': helpful_loss, 
            'helpful_acc': helpful_acc, 
            'verbosity_loss': verbosity_loss, 
            'verbosity_acc': verbosity_acc}, "./ckpt/HelpSteertest_result_verbosity_epoch_3_train.out")

end_time = time.time()
print('time:', end_time - start_time)
