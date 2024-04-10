import torch
import numpy as np
from lora import LinearLayer_LoRA
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft.tuners.tuners_utils import BaseTunerLayer

sentiment_score_model = AutoModelForSequenceClassification.from_pretrained("lvwerra/distilbert-imdb")
sentiment_score_tokenizer = AutoTokenizer.from_pretrained("lvwerra/distilbert-imdb")
def sentiment_positive_prob(prompts):
    t = sentiment_score_tokenizer(prompts, return_tensors='pt', padding=True, truncation=True)
    out = sentiment_score_model(**t)
    posi_prob = torch.softmax(out[0], dim=-1)[:, 1]
    return posi_prob

def evaluate(model, tokenizer, dataset, device, batch = 1, inds = 10):
    model.eval()
    senti_loss = [0. for _ in range(inds)]
    senti_correct = [0 for _ in range(inds)]
    brev_loss = [0. for _ in range(inds)]
    brev_correct = [0 for _ in range(inds)]
    l = 2000 #len(dataset['text'])
    dataset = dataset.shuffle()
    with torch.no_grad():
        for i in range(l // batch):
            #print(i)
            prompt = dataset['text'][i * batch : (i + 1) * batch]
            token = tokenizer(prompt,
                              return_tensors = 'pt',
                              max_length = 512,
                              padding = True,
                              truncation = True)
            senti_posi = sentiment_positive_prob(prompt)

            win_flag = []
            verbosity_metric = torch.sum(token['attention_mask'], dim=-1)
            verbosity_metric = 5 * (verbosity_metric - 266.5457) / 138.8023
            sentiment_metric = sentiment_positive_prob(prompt)
            sentiment_metric = 5 * (sentiment_metric - 0.4973) / 0.4654
            for k in range(batch // 2):
                sentiment_0 = sentiment_metric[k]
                verbosity_0 = verbosity_metric[k]
                sentiment_1 = sentiment_metric[k + batch // 2]
                verbosity_1 = verbosity_metric[k + batch // 2]
                win_flag.append([
                    sentiment_0 - sentiment_1 > 0,
                    verbosity_0 - verbosity_1 > 0
                ])
            for k, v in token.items():
                token[k] = v.to(device)

            for k in range(inds):
                force_adapter(model, adapter_names=['adapter_' + str(k),])
                model.index = k
                output = model(**token)
                p = output['probability'] # bs
                for j in range(batch // 2):
                    if win_flag[j][1]:
                        brev_loss[k] += -torch.log(p[j]).item()
                        if p[j] > 0.5:
                            brev_correct[k] += 1
                    else:
                        brev_loss[k] += -torch.log(1 - p[j]).item()
                        if p[j] < 0.5:
                            brev_correct[k] += 1

                    if win_flag[j][0]:
                        senti_loss[k] += -torch.log(p[j]).item()
                        if p[j] > 0.5:
                            senti_correct[k] += 1
                    else:
                        senti_loss[k] += -torch.log(1 - p[j]).item()
                        if p[j] < 0.5:
                            senti_correct[k] += 1

    model.train()

    senti_loss = torch.tensor(senti_loss) / (l // 2)
    senti_correct = torch.tensor(senti_correct) / (l // 2)
    brev_loss = torch.tensor(brev_loss) / (l // 2)
    brev_correct = torch.tensor(brev_correct) / (l // 2)

    return senti_loss, senti_correct, brev_loss, brev_correct

def evaluation_vector(model, tokenizer, dataset, device, batch = 1, inds = 10):
    model.eval()
    l = len(dataset['text'])
    vectors = [[] for _ in range(inds)]
    #dataset = dataset.shuffle()
    with torch.no_grad():
        for i in range(l // batch):
            #print(i)
            prompt = dataset['text'][i * batch : (i + 1) * batch]
            token = tokenizer(prompt,
                              return_tensors = 'pt',
                              max_length = 512,
                              padding = True,
                              truncation = True)
            for k, v in token.items():
                token[k] = v.to(device)
            for k in range(inds):
                force_adapter(model, adapter_names=['adapter_' + str(k),])
                model.index = k
                output = model.forward_value(**token)
                p = output['probability'] # bs
                for j in range(batch):
                    vectors[k].append(-torch.log(1 / p[j] - 1))
    model.train()
    vectors = torch.tensor(vectors)

    return vectors

def evaluate_HelpSteer(model, tokenizer, dataset, device, batch = 1, inds = 10):
    model.eval()
    helpfulness_loss = [0. for _ in range(inds)]
    helpfulness_correct = [0 for _ in range(inds)]
    verbosity_loss = [0. for _ in range(inds)]
    verbosity_correct = [0 for _ in range(inds)]
    l = min(dataset.num_rows, 2000)
    print(l)
    dataset = dataset.shuffle()
    ll = 0
    with torch.no_grad():
        for i in range(l):
            prompt = dataset['prompt'][i]
            response = dataset['response'][i]
            helpfulness_label = dataset['helpfulness'][i]
            verbosity_label = dataset['verbosity'][i]
            token = tokenizer(prompt + '[SEP]' + response,
                              return_tensors = 'pt',
                              max_length = 512,
                              padding = True,
                              truncation = True)
            str_tokens = tokenizer.convert_ids_to_tokens(token['input_ids'][0])
            if not ('[SEP]' in str_tokens):
                continue
            pos = str_tokens.index('[SEP]')
            if pos > 500:
                continue
            ll += 1
            for k, v in token.items():
                token[k] = v.to(device)
            for k in range(inds):
                LinearLayer_LoRA.index = k
                model.index = k
                output = model(**token)
                p = output['probability'] # bs
                if (helpfulness_label >= 2.5):
                    helpfulness_loss[k] += - torch.log(p[0])
                    if p[0] > 0.5:
                        helpfulness_correct[k]+= 1
                else:
                    helpfulness_loss[k] += - torch.log(1 - p[0])
                    if p[0] <= 0.5:
                        helpfulness_correct[k] += 1
                if (verbosity_label >= 1.5):
                    verbosity_loss[k] += - torch.log(p[0])
                    if p[0] > 0.5:
                        verbosity_correct[k] += 1
                else:
                    verbosity_loss[k] += - torch.log(1 - p[0])
                    if p[0] <= 0.5:
                        verbosity_correct[k] += 1

            del token, output, p
            torch.cuda.empty_cache()

    model.train()

    helpfulness_loss = torch.tensor(helpfulness_loss) / ll
    helpfulness_correct = torch.tensor(helpfulness_correct) / ll
    verbosity_loss = torch.tensor(verbosity_loss) / ll
    verbosity_correct = torch.tensor(verbosity_correct) / ll
    print(ll)

    return helpfulness_loss, helpfulness_correct, verbosity_loss, verbosity_correct

def evaluate_HelpSteer_vector(model, tokenizer, dataset, device, batch = 1, inds = 10):
    model.eval()
    l = dataset.num_rows
    dataset = dataset.shuffle()
    vectors = [[] for _ in range(inds)]
    ll = 0
    with torch.no_grad():
        for i in range(l):
            prompt = dataset['prompt'][i]
            response = dataset['response'][i]
            helpfulness_label = dataset['helpfulness'][i]
            verbosity_label = dataset['verbosity'][i]
            token = tokenizer(prompt + '[SEP]' + response,
                              return_tensors = 'pt',
                              max_length = 512,
                              padding = True,
                              truncation = True)
            str_tokens = tokenizer.convert_ids_to_tokens(token['input_ids'][0])
            if not ('[SEP]' in str_tokens):
                continue
            pos = str_tokens.index('[SEP]')
            if pos > 500:
                continue
            ll += 1
            for k, v in token.items():
                token[k] = v.to(device)
            for k in range(inds):
                LinearLayer_LoRA.index = k
                model.index = k
                output = model(**token)
                p = output['probability'] # bs
                vectors[k].append(-torch.log(1 / p[0] - 1))

    model.train()

    vectors = torch.tensor(vectors)
    print(ll)

    return vectors

def eigen_preference(num, d):
    x = torch.randint(d, size=[num,])
    x = torch.nn.functional.one_hot(x)
    return x

def gaussian_preference(num, d):
    x = torch.randint(d, size=[num,])
    x = torch.nn.functional.one_hot(x)
    y = torch.randn(size=[num, d])
    return x + y / 9

def uni_preference(num, d):
    x = torch.rand(size=[num])
    return torch.stack([x, 1 - x], dim=1)

def HelpSteer_pair_generate():
    index = np.load('temp/prompt_index_helpsteer.npy')
    l = len(index)
    p = np.max(index) + 1
    count = [0 for _ in range(p)]
    group = [[] for _ in range(p)]
    for i, x in enumerate(index):
        count[x] += 1
        group[x].append(i)
    first_responses_indices = []
    pair_map = [-1 for _ in range(l)]
    for x in range(l):
        ind = index[x]
        if count[ind] > 1:
            first_responses_indices.append(x)
            temp_group = []
            for t in group[ind]:
                if t != x :
                    temp_group.append(t)
            pair_map[x] = temp_group[np.random.randint(len(temp_group))]
    return np.array(first_responses_indices), np.array(pair_map)

def force_adapter(model, adapter_names = ['']):
    for n, m in model.named_modules():
        if isinstance(m, BaseTunerLayer):
            m._active_adapter = adapter_names