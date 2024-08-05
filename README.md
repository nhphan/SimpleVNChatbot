# Simple chatbot for Vietnamese
## A. Introduction
**Chatbot (Large Language Models):** Chatbots are one of the most popular applications in the field of Artificial Intelligence. This is increasingly evident with the advent of LLMs in recent times, which are capable of performing numerous tasks with high accuracy and are still being developed and improved. Advances in this technology not only enhance the quality of interactions between humans and machines but also open up new possibilities in automation and personalized services.

It can simulate a conversation (or a chat) with an user in natural language through messaging applications, websites, mobile apps, or through telephone.

In this repo, I will build a chatbot using LLMs trained on the Vietnamese dataset [VinaLLaMA](https://huggingface.co/vilm/vinallama-7b).
## B. Technical overview
- Pipeline:
  Download and import libraries → Load trained model → Load dataset → Create training dataset → Training model → Running prediction.

Here are the detailed steps of implementation:
### Step 1: Install libraries

```python
! pip install -q -U bitsandbytes
! pip install -q -U datasets
! pip install -q -U git+https://github.com/huggingface/transformers.git
! pip install -q -U git+https://github.com/huggingface/peft.git
! pip install -q -U git+https://github.com/huggingface/accelerate.git
! pip install -q -U loralib
! pip install -q -U einops
! pip install -q -U googletrans==3.1.0a0
```

### Step 2: Import libraries/modules

```python
import json
import os
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers

from googletrans import Translator
from pprint import pprint
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

### Step 3: Load trained model

```python
MODEL_NAME = "vilm/vinallama-7b-chat"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map = "auto",
    trust_remote_code = True,
    quantization_config = bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
```

### Step 4: Configure LLMs

```python
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "up_proj",
        "o_proj",
        "k_proj",
        "down_proj",
        "gate_proj",
        "v_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model (model, config)
generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id
```

### Step 5: Run trained model

```python
%%time
device = 'cuda' if torch.cuda.is_available() else 'cpu'

encoding = tokenizer(prompt, return_tensors ="pt").to(device)
with torch.inference_mode():
  outputs = model.generate(
    input_ids = encoding.input_ids,
    attention_mask = encoding.attention_mask,
    generation_config = generation_config
  )

print(tokenizer.decode(outputs [0], skip_special_tokens = True))
```

Following this sequence, we will be able to build a simple chatbot that can answer basic user's questions. Here are some examples of input questions and the chatbot's responses:
![image](https://github.com/nhphan/SimpleVNChatbot/assets/96032860/6a00e505-b1dd-4a27-a2e8-c3df5a8b6e60)
![image](https://github.com/nhphan/SimpleVNChatbot/assets/96032860/ec554502-b843-4ef2-a3aa-587fc37ff743)

## C. Conclusion
In conclusion, we have developed a simple chatbot that can answer basic user questions for Vietnamese. The strengths of this chatbot include its ability to understand and respond to common queries, its ease of use, and its potential to improve with more data. However, there are also some weaknesses. The chatbot may not always understand complex questions. One way to improve the model's accuracy is through fine-tuning with a specific dataset to address a particular topic. However, this repository will only utilize a general dataset.
