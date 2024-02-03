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

prompt = """
<human>: you are a medical professional, please summarise and explain this report to me in layman terms and any also follow-up lifestyle recommendations WBC (1000 cells/uL) = 3.3,Lymphocyte (%) = 31.8,Monocyte (%) = 7.8,Segmented Neutrophils (%) = 58.1,Eosinophils (%) = 1.9,Basophils (%) = 0.5,Lymphocyte (1000 cell/uL) = 3,Monocyte (1000 cell/uL) = 0.8,Segmented neutrophils (1000 cell/uL) = 3.7,Eosinophils (1000 cell/uL) = 0.1,Basophils (1000 cells/uL) = 0,RBC (million cells/uL) = 7.87,Hemoglobin (g/dL) = 13.4,Hematocrit (%) = 39.5,MCV (fL) = 81.1,MCHC (g/dl) = 33.8,MCH (pg) = 27.4,RDW (%) = 13.6,Platelet Count (1000 cells/uL) = 246,Mean Platelet Volume (fL) = 9.6
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