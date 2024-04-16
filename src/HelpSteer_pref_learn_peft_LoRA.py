import time

import os

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset
import bitsandbytes as bnb
import gc
from peft.tuners.tuners_utils import BaseTunerLayer

import wandb

from model import RewardModel
from utils import HelpSteer_pair_generate, force_adapter

inds = 2

# set device
def print_gpu_memory(flag=True):                                                            # Set False 
    if flag:
        os.system('nvidia-smi --query-gpu=memory.used --format=csv -i 2 | tail -1')
device = 'cuda:2'
device_map = {
    '': device,
}
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.float16
)

#wandb.init(
#    project = 'preference learning and tests',
#    name = 'THE NAME'                                                                       # Set the name
#)

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

model = AutoModel.from_pretrained(
    'mistralai/Mistral-7B-v0.1',                                                            # model id
    quantization_config = nf4_config,
    device_map = device_map,
    torch_dtype = torch.float16
)
#model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16, 
    lora_alpha=16, 
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

reward_model = RewardModel(tokenizer, model, inds = inds, device = device)
reward_model.v_head = reward_model.v_head.half()
reward_model.load_pretrained('ckpt/test_save')                                              # model ckpt
#reward_model.gradient_checkpointing_enable()

ori_dataset = load_dataset('nvidia/HelpSteer')
train_indices, train_pair_map = HelpSteer_pair_generate(index_file = 'temp/prompt_index_helpsteer.npy')
#eval_indices, eval_pair_map = HelpSteer_pair_generate(index_file = 'temp/prompt_index_helpsteer_val.npy')
#np.random.seed(42)
#np.random.shuffle(first_indices)

preferences = [[1.0, 0.0], 
              [np.cos(np.pi / 8), np.sin(np.pi / 8)],
              [np.cos(np.pi / 4), np.sin(np.pi / 4)],
              [np.cos(3 * np.pi / 8), np.sin(3 * np.pi / 8)],
              [0.0, 1.0]] 
weight = torch.ones([len(preferences), inds]) / inds
weight = weight.to(device)
weight.requires_grad = True
optimizer = torch.optim.Adam([weight], lr = 0.01, betas=(0.9, 0.95))

print('Start preference learning')

np.random.shuffle(train_indices)
dataset = ori_dataset['train']

# Set train batch
batch = 10

start_time = time.time()

# preference learning

for epoch in range(40):
    avg_loss = [0. for _ in range(len(preferences))]
    c = 0
    for i in range(300 // batch):
        temp_inds = train_indices[i * batch : (i + 1) * batch]
        temp_pairs = train_pair_map[temp_inds]
        sentence_input_0 = []
        sentence_input_1 = []
        win_flag = [[] for _ in range(len(preferences))]
        for k in range(batch):
            prompt = dataset['prompt'][temp_inds[k]]
            token = tokenizer(prompt,
                        padding = True,
                        truncation = True,
                        return_tensors = 'pt',
                        max_length=512)                                                     # 1024
            if torch.sum(token['attention_mask']) > 500:                                    # 1000
                continue
            c += 1
            response_0 = dataset['response'][temp_inds[k]]
            helpfulness_0 = dataset['helpfulness'][temp_inds[k]]
            verbosity_0 = dataset['verbosity'][temp_inds[k]]
            response_1 = dataset['response'][temp_pairs[k]]
            helpfulness_1 = dataset['helpfulness'][temp_pairs[k]]
            verbosity_1 = dataset['verbosity'][temp_pairs[k]]
            sentence_input_0.append(prompt + response_0)
            sentence_input_1.append(prompt + response_1)
            for l in range(len(preferences)):
                p = torch.sigmoid(torch.tensor(10) * 
                                 (preferences[l][0] * (helpfulness_0 - helpfulness_1) +
                                  preferences[l][1] * (verbosity_0 - verbosity_1)))
                win_flag[l].append(torch.rand(1).item() < p.item())
        if len(sentence_input_0) == 0:
            continue
        token = tokenizer(sentence_input_0 + sentence_input_1,
                        padding = True,
                        truncation = True,
                        return_tensors = 'pt',
                        max_length=512)                                                     # 1024
        #print(i, torch.sum(token['attention_mask'], dim=-1))
        for k, v in token.items():
            token[k] = v.to(device)
            #print(k, token[k].device, token[k].dtype)
        print_gpu_memory()
        with autocast():
            with torch.no_grad():
                chosen_scores = []
                rejected_scores = []
                for k in range(inds):
                    force_adapter(reward_model, adapter_names=['adapter_' + str(k),])
                    reward_model.index = k
                    output = reward_model(**token)
                    chosen_scores.append(output["chosen_scores"])
                    rejected_scores.append(output["rejected_scores"])
        print_gpu_memory()

        for l in range(len(preferences)):
            p = []
            for j in range(len(sentence_input_0)):
                chosen = 0
                rejected = 0
                for k in range(inds):
                    chosen += weight[l, k] * chosen_scores[k][j]
                    rejected += weight[l, k] * rejected_scores[k][j]
                p.append(torch.exp(torch.nn.functional.logsigmoid(chosen - rejected).mean()))
            loss = 0.
            for k in range(len(sentence_input_0)):
                if win_flag[l][k]:
                    loss += - torch.log(p[k])
                else:
                    loss += - torch.log(1 - p[k])
            loss.backward()
            avg_loss[l] += loss.detach().item()
        optimizer.step()
        optimizer.zero_grad()
    
    #wandb.log({
    #    'loss' + str(l): avg_loss[l] / c for l in range(len(preferences))
    #})
    
    torch.save(weight, 'ckpt/HelpSteer_weight_MODELNAME.out')                                   # save the weight, change MODELNAME

end_time = time.time()
print('time:', end_time - start_time)

# wandb.finish()