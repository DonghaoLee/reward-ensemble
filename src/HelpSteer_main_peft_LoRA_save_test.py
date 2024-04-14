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

for ind in range(inds):
    model.add_adapter(config, 'adapter_' + str(ind))
model.set_adapter(['adapter_' + str(ind) for ind in range(inds)])
model.gradient_checkpointing_enable()
for n, m in model.named_modules():
    if isinstance(m, BaseTunerLayer):
        m = m.half()
reward_model = RewardModel(tokenizer, model, inds = inds, device = device)
reward_model.v_head = reward_model.v_head.half()
# check the parameters if necessary
#for n, p in reward_model.named_parameters():
#    print(n, p.dtype, p.requires_grad, p.device)

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
reward_model.save_pretrained('ckpt/test_save')

end_time = time.time()
print('time:', end_time - start_time)

#wandb.finish()