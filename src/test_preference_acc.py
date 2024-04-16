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
    prepare_model_for_kbit_training, 
    LoraConfig, 
    get_peft_model
)
from peft.tuners.tuners_utils import BaseTunerLayer
from datasets import load_dataset

import wandb

from model import RewardModel
from utils import force_adapter

inds = 2
#torch.set_default_dtype(torch.float16)
# set device
device = 'cuda:1'
device_map = {
    '': device,
}

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

reward_model.load_state_dict(torch.load("./ckpt/IMDB_LoRA_EM_epoch_4_uni.ckpt", map_location=device_map))

o_dataset = load_dataset('imdb', split="test")
len_dataset = o_dataset.num_rows

#weight = [torch.load('ckpt/reward_0.5_w_1.0_0.0_weight.out').to(device),
#          torch.load('ckpt/reward_0.5_w_0.98_0.38_weight.out').to(device),
#          torch.load('ckpt/reward_0.5_w_0.53_0.53_weight.out').to(device),
#          torch.load('ckpt/reward_0.5_w_0.38_0.98_weight.out').to(device),
#          torch.load('ckpt/reward_0.5_w_0.0_1.0_weight.out').to(device)]
weight = torch.load('ckpt/reward_uni_weight.out')

start_time = time.time()
batch = 200

dataset = o_dataset.shuffle().select(range(2000))
preferences = [[1.0, 0.0], 
              [np.cos(np.pi / 8), np.sin(np.pi / 8)],
              [np.cos(np.pi / 4), np.sin(np.pi / 4)],
              [np.cos(3 * np.pi / 8), np.sin(3 * np.pi / 8)],
              [0.0, 1.0]] 
#[np.cos(np.pi / 4), np.sin(np.pi / 4)]

correct_count = np.zeros(5)
avg_loss = np.zeros(5)
for i in range(len(dataset['text']) // batch):
    prompt = dataset['text'][i * batch : (i + 1) * batch]
    # label = dataset['label'][i * batch : (i + 1) * batch]
    token = tokenizer(prompt,
                      padding = True,
                      truncation = True,
                      return_tensors = 'pt',
                      max_length=512)
    verbosity_metric = torch.sum(token['attention_mask'], dim=-1)
    verbosity_metric = 5 * (verbosity_metric - 266.5457) / 138.8023
    sentiment_metric = sentiment_positive_prob(prompt)
    sentiment_metric = 5 * (sentiment_metric - 0.4973) / 0.4654
    win_flag = []
    for k in range(batch // 2):
        sentiment_0 = sentiment_metric[k]
        verbosity_0 = verbosity_metric[k]
        sentiment_1 = sentiment_metric[k + batch // 2]
        verbosity_1 = verbosity_metric[k + batch // 2]
        flag = [((preferences[l][0] * (sentiment_0 - sentiment_1) +
                    preferences[l][1] * (verbosity_0 - verbosity_1)) > 0.5) for l in range(5)]
        win_flag.append(flag)
    for k, v in token.items():
        token[k] = v.to(device)
    with torch.no_grad():
        chosen_scores = []
        rejected_scores = []
        for k in range(inds):
            force_adapter(reward_model, adapter_names=['adapter_' + str(k),])
            reward_model.index = k
            output = reward_model(**token)
            chosen_scores.append(output["chosen_scores"])
            rejected_scores.append(output["rejected_scores"])
        p = []
        for j in range(batch // 2):
            dif = [0 for l in range(5)]
            for k in range(inds):
                for l in range(5):
                    dif[l] += weight[l][k] * (chosen_scores[k][j] - rejected_scores[k][j])
            p.append([torch.exp(torch.nn.functional.logsigmoid(dif[l]).mean()) for l in range(5)])
    loss = np.zeros(5)
    for l in range(5):
        for k in range(batch // 2):
            if win_flag[k][l]:
                loss[l] += - torch.log(p[k][l])
                if p[k][l] > 0.5:
                    correct_count[l] += 1
            else:
                loss[l] += - torch.log(1 - p[k][l])
                if p[k][l] < 0.5:
                    correct_count[l] += 1
        loss[l] = loss[l] / (batch // 2)
        avg_loss[l] += loss[l].item()
avg_loss = avg_loss / 10.
acc = correct_count / 1000.

print(acc, avg_loss)
torch.save(
    {'acc': acc, 'loss': loss, 'pref':preferences},
    'ckpt/pref_result_uni_eigen.out'
)

end_time = time.time()
print('time:', end_time - start_time)