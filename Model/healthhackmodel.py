#pip install -Uqqq pip
#pip install -qqq bitsandbytes
#pip install -qqq torch
#pip install -qqq -U git+https://github.com/huggingface/transformers.git@e03a9cc
#pip install -qqq -U git+https://github.com/huggingface/peft.git@42a184f
#pip install -qqq -U git+https://github.com/huggingface/accelerate.git@c9fbb71
#pip install -qqq datasets
#pip install -qqq loralib
#pip install -qqq einops
#nvcc --version
#pip install --upgrade tensorflow

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch
from huggingface_hub import login, logout
import config

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

PEFT_MODEL = 'healthhack-trained-model'

login(config.access_token) # login with your own huggingface token, note that our base model is the llama-2-7b-chat-hf model, so you will need to request for
                           # access on meta and huggingface.
config = PeftConfig.from_pretrained(PEFT_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(model, PEFT_MODEL)
logout()