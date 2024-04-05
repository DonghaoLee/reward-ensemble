import time

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset

import wandb

from model import RewardModel
from utils import HelpSteer_pair_generate, force_adapter

inds = 2
torch.set_default_dtype(torch.float16)
# set device
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
)
model = prepare_model_for_kbit_training(model)

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
reward_model = RewardModel(tokenizer, model, inds = inds, device = device)
for n, p in reward_model.named_parameters():
    print(n, p.dtype, p.requires_grad, p.device)

dataset = load_dataset('nvidia/HelpSteer', split="train")
first_indices, pair_map = HelpSteer_pair_generate()

optimizer = torch.optim.Adam(reward_model.parameters(), lr = 0.00001, betas=(0.9, 0.95))

start_time = time.time()
batch = 1
ratio = 0.5

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
                          max_length=500) # 1024
            if torch.sum(token['attention_mask']) > 400: # 1000
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
                          max_length=500) #
        #print(i, torch.sum(token['attention_mask'], dim=-1))
        for k, v in token.items():
            token[k] = v.to(device)
            print(k, token[k].device, token[k].dtype)

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
        print(loss)
        #wandb.log({
        #    'loss': loss.item(),
        #})
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        flag_list = torch.tensor(flag_list).to(device)

        for k in range(inds):
            force_adapter(reward_model, adapter_names=['adapter_' + str(k),])
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

#wandb.finish()