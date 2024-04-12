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

inds = 10
#torch.set_default_dtype(torch.float16)
# set device
device = 'cuda:2'
device_map = {
    '': device,
}

wandb.init(
    project = 'preference learning',
    name = 'from uni_epoch_4 learn (0.53, 0.53)'
)

sentiment_score_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-350m')
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True

sentiment_score_model = AutoModelForSequenceClassification.from_pretrained(
    "lvwerra/distilbert-imdb",
    device_map = device_map,
    #torch_dtype=torch.float16
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

reward_model.load_state_dict(torch.load("./ckpt/IMDB_LoRA_Ensemble_0.5_epoch_4_uni.ckpt", map_location=device_map))
#print('reward model')
#for n, p in reward_model.named_parameters():
#    print(n, p.dtype, p.requires_grad, p.device)
#print()

o_dataset = load_dataset('imdb', split="train")
len_dataset = o_dataset.num_rows

weight = torch.ones(inds) / inds
weight = weight.to(device)
weight.requires_grad = True
optimizer = torch.optim.Adam([weight], lr = 0.01, betas=(0.9, 0.95))

start_time = time.time()
batch = 40

dataset = o_dataset.shuffle().select(range(200))

#[np.cos(np.pi / 4), np.sin(np.pi / 4)]
preference = [np.cos(np.pi / 4), np.sin(np.pi / 4)] 

for epoch in range(40): # 2 epochs
    # It will be a better idea to use DatasetLoader in Pytorch to load the data
    # Here I just use the shuffle.
    # dataset = dataset.shuffle()
    avg_loss = 0.
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
            p = torch.sigmoid(preference[0] * (sentiment_0 - sentiment_1) +
                              preference[1] * (verbosity_0 - verbosity_1))
            win_flag.append(torch.rand(1).item() < p.item())

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
            chosen = 0
            rejected = 0
            for k in range(inds):
                chosen += weight[k] * chosen_scores[k][j]
                rejected += weight[k] * rejected_scores[k][j]
            p.append(torch.exp(torch.nn.functional.logsigmoid(chosen - rejected).mean()))
        loss = 0.
        for k in range(batch // 2):
            if win_flag[k]:
                loss += - torch.log(p[k])
            else:
                loss += - torch.log(1 - p[k])
        loss = loss / (batch // 2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss += loss.detach().item()
    avg_loss = avg_loss / 5.
    wandb.log({
        'loss': avg_loss,
    })

    # torch.save(reward_model.state_dict(), 'ckpt/mix_reward_model_0.5_epoch_' + str(epoch) + '.ckpt')
    torch.save(weight, 'ckpt/reward_uni_w_0.53_0.53_weight.out')

end_time = time.time()
print('time:', end_time - start_time)

wandb.finish()
