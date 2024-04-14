import time

import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForSequenceClassification, 
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model
)
from datasets import load_dataset

import wandb

from model import RewardModel
from utils import force_adapter

inds = 2
# set device
device = 'cuda:1'
device_map = {
    '': device,
}

wandb.init(
    project = 'ensemble reward model with LoRA',
    name = 'training LoRA EM - IMDB - uni'
)

sentiment_score_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m')
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

sentiment_score_model = AutoModelForSequenceClassification.from_pretrained(
    "lvwerra/distilbert-imdb",
    device_map = device_map,
)
def sentiment_positive_prob(prompts):
    with torch.no_grad():
        t = sentiment_score_tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(device)
        out = sentiment_score_model(**t)
        posi_prob = torch.softmax(out[0], dim=-1)[:, 1].to('cpu')
    return posi_prob

model = AutoModel.from_pretrained(
    'facebook/opt-350m',
    device_map = device_map,
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
#print('reward model')
#for n, p in reward_model.named_parameters():
#    print(n, p.dtype, p.requires_grad, p.device)
#print()
#reward_model.load_state_dict(torch.load("./ckpt/IMDB_LoRA_Ensemble_0.5_epoch_4.ckpt", map_location=device))

o_dataset = load_dataset('imdb', split="train")
len_dataset = o_dataset.num_rows
print(len_dataset)
preferences = torch.load('preferences/preference_d2_0.pt')
preferences = preferences['uni']

learned_pref = torch.randint(2, [len(preferences),])

optimizer = torch.optim.Adam(reward_model.parameters(), lr = 0.00001, betas=(0.9, 0.95))

start_time = time.time()
batch = 10

original_indices = np.arange(len_dataset)
np.random.shuffle(original_indices)
preferences_shuffled = preferences[torch.tensor(original_indices)]
dataset = o_dataset.select(original_indices)

for epoch in range(5): # epochs
    for i in range(len(dataset['text']) // batch):
        #print(10 * '-' + 'i = ' + str(i) + 10 * '-')
        win_flag = []
        
        prompt = dataset['text'][i * batch : (i + 1) * batch]
        indices = original_indices[i * batch : (i + 1) * batch]
        token = tokenizer(prompt,
                          padding = True,
                          truncation = True,
                          return_tensors = 'pt',
                          max_length=512)
        #print(i, torch.sum(token['attention_mask'], dim=-1))
        verbosity_metric = torch.sum(token['attention_mask'], dim=-1)
        verbosity_metric = 5 * (verbosity_metric - 266.5457) / 138.8023
        sentiment_metric = sentiment_positive_prob(prompt)
        sentiment_metric = 5 * (sentiment_metric - 0.4973) / 0.4654
        temp_preferences = preferences_shuffled[i * batch : (i + 1) * batch]
        for k in range(batch // 2):
            sentiment_0 = sentiment_metric[k]
            verbosity_0 = verbosity_metric[k]
            sentiment_1 = sentiment_metric[k + batch // 2]
            verbosity_1 = verbosity_metric[k + batch // 2]
            p = torch.sigmoid(temp_preferences[k, 0] * (sentiment_0 - sentiment_1) +
                              temp_preferences[k, 1] * (verbosity_0 - verbosity_1))
            win_flag.append(torch.rand(1).item() < p.item())
        win_flag = torch.tensor(win_flag).to(device)

        for k, v in token.items():
            token[k] = v.to(device)

        with torch.no_grad():
            p = []
            for k in range(inds):
                force_adapter(reward_model, adapter_names=['adapter_' + str(k),])
                reward_model.index = k
                output = reward_model(**token)
                p.append(output["probability"])
                #print('adapter' + str(k), output["probability"])
            p = torch.stack(p, dim=1) # bs, inds
            adapter_inds = [[] for _ in range(inds)]
            for k in range(batch // 2):
                if win_flag[k]:
                    learned_pref[indices[k]] = torch.argmax(p[k])
                else:
                    learned_pref[indices[k]] = torch.argmin(p[k])
                adapter_inds[learned_pref[indices[k]]].append(k)

        total_loss = 0.
        for k in range(inds):
            if len(adapter_inds[k]) > 0:
                loss = 0.
                #reward_model.rwtransformer.set_adapter('adapter_' + str(k))
                force_adapter(reward_model, adapter_names=['adapter_' + str(k),])
                reward_model.index = k
                output = reward_model(**token)
                p = output["probability"]
                temp = win_flag * p + (1 - win_flag.int()) * (1 - p) # bs
                loss = -torch.log(temp[torch.tensor(adapter_inds[k])]).mean()
                loss.backward()
                total_loss += loss.item() * len(adapter_inds[k])
        wandb.log({
            'loss': total_loss / (batch // 2)
        })
        optimizer.step()
        optimizer.zero_grad()
    torch.save(reward_model.state_dict(), 'ckpt/IMDB_LoRA_EM_epoch_' + str(epoch) + '_uni.ckpt')

end_time = time.time()
print('time:', end_time - start_time)

wandb.finish()