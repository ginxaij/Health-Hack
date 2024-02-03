from transformers import AutoTokenizer
from healthhackmodel import model
from peft import PeftConfig
import torch

PEFT_MODEL = 'healthhack-trained-model/'
config = PeftConfig.from_pretrained(PEFT_MODEL)

tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

generation_config = model.generation_config
generation_config.max_new_tokens = 1300
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

#%%time
device = "cuda:0"

input = ""
prompt = f"""
<human>: you are an ai medical chatbot used to give advice and recommendations based on a patient's medical symptoms.  Please generate a response according to this input: {input}
<assistant>:
""".strip()

encoding = tokenizer(prompt, return_tensors="pt").to(device)
with torch.inference_mode():
  outputs = model.generate(
      input_ids = encoding.input_ids,
      attention_mask = encoding.attention_mask,
      generation_config = generation_config
  )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))