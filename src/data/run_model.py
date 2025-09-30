import torch
from transformers import AutoModel, AutoTokenizer
import os
import sys
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == "__main__" or parent_module.__name__ == '__main__':
    from token_splits import pretokenizer_dict
else:
    from .token_splits import pretokenizer_dict
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_dir = os.getcwd()
m_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "models/lms/lms")
tokenizer_path = os.path.join(m_path, "tok_lm")
models_path = os.path.join(m_path, "mol_lm")

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