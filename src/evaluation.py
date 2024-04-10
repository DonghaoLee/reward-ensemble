import time

import torch
import torch.nn as nn
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel
from peft import (
    prepare_model_for_kbit_training, 
    LoraConfig, 
    get_peft_model
)
from datasets import load_dataset

import wandb

from model import RewardModel
from utils import evaluate

inds = 10
device = 'cuda:2'
device_map = {
    '': device,
}

# load dataset
imdb_dataset = load_dataset("imdb", split='test')

# load tokenizer. It will embed the input sentence.
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m',
                                          padding_side = 'right',
                                          truncation_side = 'right')

# load model
# load model
model = AutoModel.from_pretrained(
    'facebook/opt-350m',
    device_map = device_map,
    #torch_dtype=torch.float16
)
config = LoraConfig(
    r=32, 
    lora_alpha=1, 
    target_modules=['k_proj', 'q_proj', 'v_proj', 'out_proj', "fc1", "fc2"], 
    lora_dropout=0.00, 
    bias="none", 
    task_type="CAUSAL_LM"
)
for ind in range(inds):
    model.add_adapter(config, 'adapter_' + str(ind))
model.set_adapter(['adapter_' + str(ind) for ind in range(inds)])
reward_model = RewardModel(
    tokenizer, 
    model, 
    inds = inds, 
    device = device, 
    num_padding_at_beginning=1
)
reward_model.load_state_dict(torch.load("./ckpt/IMDB_LoRA_Ensemble_0.5_epoch_4.ckpt", map_location=device_map))

start_time = time.time()

print(torch.softmax(reward_model.weight, dim=0))
senti_loss, senti_acc, brev_loss, brev_acc = evaluate(reward_model, tokenizer,
                                             imdb_dataset, device, batch=100)
torch.save({'senti_loss': senti_loss, 
            'senti_acc': senti_acc, 
            'brev_loss': brev_loss, 
            'brev_acc': brev_acc}, "./ckpt/test_result_IMDB_epoch_4.out")

end_time = time.time()
print('time:', end_time - start_time)
