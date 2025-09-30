from transformers import AutoModel, AutoTokenizer

from token_splits import pretokenizer_dict
import json
import os
import torch
import re

current_dir = os.getcwd()
m_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))), "models/lms/lms")
tokenizer_path = os.path.join(m_path, "tok_lm")
models_path = os.path.join(m_path, "mol_lm")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_model(device):
    tokenizer_type = 'simple_regex'
    with open(tokenizer_path + "/config.json", "r") as f:
        tokenizer_config = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, **tokenizer_config
    )
    tokenizer.backend_tokenizer.pre_tokenizer = pretokenizer_dict[
        tokenizer_type
    ]

    # Load the pre-trained model
    model = AutoModel.from_pretrained(models_path).to(device)

    return tokenizer, model

def tokenize(tokenizer, input_text, device):
    tokenized_input = tokenizer(input_text, return_tensors="pt")
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

    return tokenized_input

def get_embedding(model, tokenized_input):
    with torch.no_grad():
        return model(**tokenized_input).last_hidden_state
    

tokenizer, model = load_model(device) 


input_text = "C1C[C@H]2C34OC23[C@H]14"
print(f"Pretokenizer length: {len(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(input_text))}")

print(f"Input length: {len(input_text)}")

tokenized_input = tokenize(tokenizer, input_text, device)

print(tokenized_input['input_ids'].size())

# Move the input tensor to the appropriate device (e.g., GPU) if necessary

# Call the model with the tokenized input
with torch.no_grad():
    model_output = model(**tokenized_input)
    print(f"Embedding shape: {model_output.last_hidden_state.size()}")

