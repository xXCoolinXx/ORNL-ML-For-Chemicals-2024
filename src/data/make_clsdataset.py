import sys
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == "__main__" or parent_module.__name__ == '__main__':
    from download_customqm9 import CustomQM9
    from run_model import *
else:
    from .download_customqm9 import CustomQM9
    from .run_model import *
import torch
from torch.utils.data import Dataset
 
from tqdm import tqdm

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data/custom_qm9")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QM9CLS(Dataset):
    def __init__(self, save_dir):
        super().__init__()
        self.embeddings = None
        self.save_dir = save_dir
        self.generate_embeddings()

    def generate_embeddings(self):
        save_path = os.path.join(self.save_dir, 'cls_embeddings.pt')
        if os.path.exists(save_path):
            print("Loading embeddings from cached file...")
            self.embeddings = torch.load(save_path)
        else:
            print("Generating embeddings...")
            cls_list = []
            tokenizer, model = load_model(device)
            molecules = CustomQM9(root = data_dir)

            for mol in tqdm(molecules):
                embedding = get_embedding(model, tokenize(tokenizer, mol.smiles, device))
                cls_list.append(embedding[0][0].to(torch.device("cpu")))  # Get cls embedding, have to move to the CPU to avoid OOM issues
            
            self.embeddings = cls_list

            os.makedirs(self.save_dir, exist_ok=True)
            torch.save(self.embeddings, save_path)

    def __len__(self):
        if self.embeddings is None:
            raise ValueError("Embeddings not generated. Call 'generate_embeddings' first.")
        return len(self.embeddings)

    def __getitem__(self, idx):
        if self.embeddings is None:
            raise ValueError("Embeddings not generated. Call 'generate_embeddings' first.")
        return self.embeddings[idx]
    
if __name__ == "__main__": 
    QM9CLS("../../data/etc/")