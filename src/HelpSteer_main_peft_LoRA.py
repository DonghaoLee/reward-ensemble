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

inds = 1
#torch.set_default_dtype(torch.float16)

# set device
def print_gpu_memory():
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
#    project = 'mix reward model with LoRA',
#    name = ''
#)

tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

model = AutoModel.from_pretrained(
    'mistralai/Mistral-7B-v0.1',
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

for ind in range(inds):
    model.add_adapter(config, 'adapter_' + str(ind))
model.set_adapter(['adapter_' + str(ind) for ind in range(inds)])
model.gradient_checkpointing_enable()
for n, m in model.named_modules():
    if isinstance(m, BaseTunerLayer):
        m = m.half()
reward_model = RewardModel(tokenizer, model, inds = inds, device = device)
# check the parameters if necessary
for n, p in reward_model.named_parameters():
    print(n, p.dtype, p.requires_grad, p.device)

dataset = load_dataset('nvidia/HelpSteer', split="train")
first_indices, pair_map = HelpSteer_pair_generate()

print_gpu_memory()

params = []
for n,p in reward_model.named_parameters():
    if n != 'weight':
        params.append(p)
optimizer_1 = bnb.optim.PagedAdam8bit(
    params, 
    lr = 0.00001, 
    betas=(0.9, 0.95)
)
optimizer_2 = torch.optim.Adam([reward_model.weight], lr = 0.00001, betas=(0.9, 0.95))
#optimizer = torch.optim.Adam(reward_model.parameters(), lr = 0.00001, betas=(0.9, 0.95))
scaler = GradScaler()

start_time = time.time()
batch = 1
ratio = 0.5

np.random.seed(42)
np.random.shuffle(first_indices)

for epoch in range(5): # epochs
    for i in range(len(first_indices) // batch):
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
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
                          max_length=512) # 1024
            if torch.sum(token['attention_mask']) > 500: # 1000
                continue
            response_0 = dataset['response'][temp_inds[k]]
            helpfulness_0 = dataset['helpfulness'][temp_inds[k]]
            verbosity_0 = dataset['verbosity'][temp_inds[k]]
            response_1 = dataset['response'][temp_pairs[k]]
            helpfulness_1 = dataset['helpfulness'][temp_pairs[k]]
            verbosity_1 = dataset['verbosity'][temp_pairs[k]]
            sentence_input_0.append(prompt + response_0)
            sentence_input_1.append(prompt + response_1)
            p_helpfulness = torch.sigmoid(10 * torch.tensor(helpfulness_0 - helpfulness_1))
            p_verbosity = torch.sigmoid(10 * torch.tensor(verbosity_0 - verbosity_1))
            win_flag.append([torch.rand(1).item() < p_helpfulness, torch.rand(1).item() < p_verbosity])

        if len(sentence_input_0) == 0:
            continue
        token = tokenizer(sentence_input_0 + sentence_input_1,
                          padding = True,
                          truncation = True,
                          return_tensors = 'pt',
                          max_length=512) 

        print(i, torch.sum(token['attention_mask'], dim=-1))

        for k, v in token.items():
            token[k] = v.to(device)
            print(k, token[k].device, token[k].dtype)

        print_gpu_memory()

        with autocast():
            with torch.no_grad():
                p = []
                for k in range(inds):
                    force_adapter(reward_model, adapter_names=['adapter_' + str(k),])
                    reward_model.index = k
                    output = reward_model(**token)
                    p.append(output["probability"])
                base_probs = torch.stack(p, dim=1) # bs, inds=10
                w = torch.softmax(reward_model.weight, dim=0) # inds
                base_prob = torch.matmul(base_probs, w) # bs
            
                print_gpu_memory()
        base_prob = base_prob.float()
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
        loss = loss / len(sentence_input_0)
        #wandb.log({
        #    'loss': loss.item(),
        #})
        loss.backward()
        optimizer_2.step()

        #scaler.scale(loss).backward()
        flag_list = torch.tensor(flag_list).to(device)

        for k in range(inds):
            print(k)
            force_adapter(reward_model, adapter_names=['adapter_' + str(k),])
            reward_model.index = k
            with autocast():
                output = reward_model(**token)
                p = output["probability"]
                print(p.dtype)
                loss = torch.mean(w[k] * p / (flag_list - base_prob))
            print_gpu_memory()
            #scaler.scale(loss).backward()
            loss.backward()

        optimizer_1.step()
        #scaler.step(optimizer_1)
        #scaler.update()

        del loss, token, p, flag_list
        torch.cuda.empty_cache()
        gc.collect()
    
    break

    torch.save(reward_model.state_dict(), 'ckpt/HelpSteer_mix_0.5_epoch_' + str(epoch) + '.ckpt')

end_time = time.time()
print('time:', end_time - start_time)

#wandb.finish()