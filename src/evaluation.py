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
from utils import evaluate

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
device = "cuda:0"

# load a reward model
reward_model = RewardModel(tokenizer, model)
convert_linear_layer_to_lora(reward_model, "layers.", lora_dim = 64, inds = 10)
reward_model.load_state_dict(torch.load("./ckpt/mix_reward_model_0.5_epoch_9.ckpt"))
reward_model = reward_model.to(device)

start_time = time.time()

print(torch.softmax(reward_model.weight, dim=0))
senti_loss, senti_acc, brev_loss, brev_acc = evaluate(reward_model, tokenizer,
                                             imdb_dataset['test'], device, batch=100)
torch.save({'senti_loss': senti_loss, 
            'senti_acc': senti_acc, 
            'brev_loss': brev_loss, 
            'brev_acc': brev_acc}, "./ckpt/test_result_0.5_epoch_9.out")

end_time = time.time()
print('time:', end_time - start_time)
