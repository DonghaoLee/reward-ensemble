import torch
from lora import LinearLayer_LoRA

def evaluate(model, tokenizer, dataset, device, batch = 1, inds = 10):
    model.eval()
    senti_loss = [0. for _ in range(inds)]
    senti_correct = [0 for _ in range(inds)]
    brev_loss = [0. for _ in range(inds)]
    brev_correct = [0 for _ in range(inds)]
    dataset = dataset.shuffle()
    l = 1000
    with torch.no_grad():
        for i in range(l // batch):
            #print(i)
            prompt = dataset['text'][i * batch : (i + 1) * batch]
            label = dataset['label'][i * batch : (i + 1) * batch]
            token = tokenizer(prompt,
                              return_tensors = 'pt',
                              max_length = 512,
                              padding = True,
                              truncation = True)
            for k, v in token.items():
                token[k] = v.to(device)
            for k in range(inds):
                LinearLayer_LoRA.index = k
                model.index = k
                output = model(**token)
                p = output['probability'] # bs
                for j in range(batch):
                    ll = torch.sum(token['attention_mask'][j]).cpu()
                    p_brevity = torch.sigmoid((ll - 20) / 10) * torch.sigmoid((150 - ll) / 40) * 1.2

                    if torch.rand(1) < p_brevity:
                        brev_loss[k] += -torch.log(p[j]).item()
                        if p[j] > 0.5:
                            brev_correct[k] += 1
                    else:
                        brev_loss[k] += -torch.log(1 - p[j]).item()
                        if p[j] < 0.5:
                            brev_correct[k] += 1

                    if label[j] == 1:
                        senti_loss[k] += -torch.log(p[j]).item()
                        if p[j] > 0.5:
                            senti_correct[k] += 1
                    else:
                        senti_loss[k] += -torch.log(1 - p[j]).item()
                        if p[j] < 0.5:
                            senti_correct[k] += 1

            del token, output, p
            torch.cuda.empty_cache()

    model.train()

    senti_loss = torch.tensor(senti_loss) / l
    senti_correct = torch.tensor(senti_correct) / l
    brev_loss = torch.tensor(brev_loss) / l
    brev_correct = torch.tensor(brev_correct) / l

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
            label = dataset['label'][i * batch : (i + 1) * batch]
            token = tokenizer(prompt,
                              return_tensors = 'pt',
                              max_length = 512,
                              padding = True,
                              truncation = True)
            for k, v in token.items():
                token[k] = v.to(device)
            for k in range(inds):
                LinearLayer_LoRA.index = k
                model.index = k
                output = model.forward_value(**token)
                p = output['probability'] # bs
                for j in range(batch):
                    vectors[k].append(-torch.log(1 / p[j] - 1))

            del token, output, p
            torch.cuda.empty_cache()

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
