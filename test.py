import time
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

torch.set_default_dtype(torch.bfloat16)
device = 'cuda:0'
device_map = {
    '': device,
}
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

pretrained_model_path = "mistralai/Mistral-7B-v0.1"
model = AutoModel.from_pretrained(
    pretrained_model_path,
    quantization_config=nf4_config,
    device_map = device_map,
    torch_dtype = torch.bfloat16
)
for n, p in model.named_parameters():
    print(n, p.dtype, p.device, p.requires_grad)


print('sleep')
time.sleep(30)
print('end')