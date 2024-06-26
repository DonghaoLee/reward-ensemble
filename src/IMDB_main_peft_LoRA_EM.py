import time

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import load_dataset

import wandb

from model import RewardModel

inds = 10
torch.set_default_dtype(torch.float16)
# set device
device = 'cuda:1'
device_map = {
    '': device,
}

wandb.init(
    project = 'ensemble reward model with LoRA',
    name = 'training LoRA ensemble - IMDB'
)

sentiment_score_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m')
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

sentiment_score_model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
def sentiment_positive_prob(prompts):
    t = sentiment_score_tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    out = sentiment_score_model(**t)
    posi_prob = torch.softmax(out[0], dim=-1)[:, 1]
    return posi_prob

model = AutoModel.from_pretrained(
    'facebook/opt-350m',
    device_map = device_map
)
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=16, 
    lora_alpha=16, 
    target_modules=['k_proj', 'q_proj', 'v_proj', 'out_proj', "fc1", "fc2"], 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)

for ind in range(inds):
    model.add_adapter(config, 'adapter_' + str(ind))
model.set_adapter('adapter_0')
reward_model = RewardModel(tokenizer, model, inds = inds, device = device)
for n, p in reward_model.named_parameters():
    print(n, p.dtype, p.requires_grad, p.device)

dataset = load_dataset('imdb', split="train")

optimizer = torch.optim.Adam(reward_model.parameters(), lr = 0.00001, betas=(0.9, 0.95))

start_time = time.time()
batch = 2
ratio = 0.5

for epoch in range(5): # epochs
    #dataset.shuffle()
    for i in range(len(dataset['text']) // batch):
        win_flag = []
        
        prompt = dataset['text'][i * batch : (i + 1) * batch]
        token = tokenizer(sentence_input_0 + sentence_input_1,
                          padding = True,
                          truncation = True,
                          return_tensors = 'pt',
                          max_length=1024)
        #print(i, torch.sum(token['attention_mask'], dim=-1))
        verbosity_metric = torch.sum(token['attention_mask'], dim=-1)
        verbosity_metric = 5 * (verbosity_metric - 266.5457) / 138.8023
        sentiment_metric = sentiment_positive_prob(prompt)
        sentiment_metric = 5 * (sentiment_metric - 0.4973) / 0.4654
        for k in range(batch // 2):
            sentiment_0 = sentiment_metric[k]
            verbosity_0 = verbosity_metric[k]
            sentiment_1 = sentiment_metric[k + batch // 2]
            verbosity_1 = verbosity_metric[k + batch // 2]
            p_sentiment = torch.sigmoid(torch.tensor(helpfulness_0 - helpfulness_1))
            p_verbosity = torch.sigmoid(torch.tensor(verbosity_0 - verbosity_1))
            win_flag.append([torch.rand(1).item() < p_sentiment, torch.rand(1).item() < p_verbosity])

        for k, v in token.items():
            token[k] = v.to(device)

        with torch.no_grad():
            p = []
            for k in range(inds):
                reward_model.rwtransformer.set_adapter('adapter_' + str(k))
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
        loss = loss / (batch // 2)
        #wandb.log({
        #    'loss': loss.item(),
        #})
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        flag_list = torch.tensor(flag_list).to(device)

        for k in range(inds):
            reward_model.rwtransformer.set_adapter('adapter_' + str(k))
            reward_model.index = k
            output = reward_model(**token)
            p = output["probability"]
            loss = torch.mean(w[k] * p / (flag_list - base_prob))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    torch.save(reward_model.state_dict(), 'ckpt/IMDB_LoRA_Ensemble_0.5_epoch_' + str(epoch) + '.ckpt')

end_time = time.time()
print('time:', end_time - start_time)

#wandb.finish()