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
#torch.set_default_dtype(torch.float16)

# set device
def print_gpu_memory(flag=True):
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

reward_model = RewardModel(tokenizer, model, inds = inds, device = device)
reward_model.v_head = reward_model.v_head.half()
reward_model.load_pretrained('ckpt/test_save')
#reward_model.gradient_checkpointing_enable()

start_time = time.time()
batch = 1

dataset = load_dataset('nvidia/HelpSteer', split="validation")
first_indices, pair_map = HelpSteer_pair_generate(index_file = 'temp/prompt_index_helpsteer_val.npy')
#np.random.seed(42)
#np.random.shuffle(first_indices)

loss = torch.zeros(2, inds)
count = torch.zeros(2, inds)
c = 0
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
                      max_length=512)                       # 1024
        if torch.sum(token['attention_mask']) > 500:        # 1000
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
        win_flag.append([0.5 < p_helpfulness, 0.5 < p_verbosity])
    if len(sentence_input_0) == 0:
        continue
    token = tokenizer(sentence_input_0 + sentence_input_1,
                      padding = True,
                      truncation = True,
                      return_tensors = 'pt',
                      max_length=512)                       # 1024
    #print(i, torch.sum(token['attention_mask'], dim=-1))
    for k, v in token.items():
        token[k] = v.to(device)
        #print(k, token[k].device, token[k].dtype)
    print_gpu_memory()
    with autocast():
        with torch.no_grad():
            for k in range(inds):
                force_adapter(reward_model, adapter_names=['adapter_' + str(k),])
                reward_model.index = k
                output = reward_model(**token)
                for l, p in enumerate(output["probability"]):
                    p = p.cpu()
                    print(p)
                    c += 1
                    for v in range(2):
                        if win_flag[l][v]:
                            loss[v, k] += - torch.log(p)
                            if p > 0.5:
                                count[v, k] += 1
                        else:
                            loss[v, k] += - torch.log(1 - p)
                            if p <= 0.5:
                                count[v, k] += 1
    print_gpu_memory()

torch.save({
    'loss': loss / c,
    'acc': count / c
}, 'two_reward_acc_test.out') # model_name + '_two_reward_acc_test.out'

end_time = time.time()
print('time:', end_time - start_time)