import numpy as np
import torch
from datasets import load_dataset
from utils import eigen_preference, gaussian_preference, uni_preference

HelpSteer = load_dataset('nvidia/HelpSteer')

l = HelpSteer['train'].num_rows
d = 2

w1 = eigen_preference(l, d)
w2 = gaussian_preference(l, d)
w3 = uni_preference(l, d)

torch.save({
    'eigen': w1,
    'gaussian': w2,
    'uni': w3
}, 'preferences/preference_d2_0.pt')
