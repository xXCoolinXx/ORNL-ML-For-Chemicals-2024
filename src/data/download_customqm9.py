import torch_geometric as tg
from torch_geometric.datasets.qm9 import QM9
import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import re
from torch.nn.functional import one_hot
from tqdm import tqdm
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import one_hot, scatter
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset
import gc 

#Stupid python stuff, this shouldn't exist
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == "__main__" or parent_module.__name__ == '__main__':
    from token_splits import pretokenizer_dict
    from run_model import *
else:
    from .token_splits import pretokenizer_dict
    from .run_model import *

import json
from torch_geometric.datasets.qm9 import conversion

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "data/custom_qm9")

class CustomQM9(QM9):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Call the parent's __init__ method

    def process(self):
        
        tokenizer, model = load_model(device)

        import rdkit
        from rdkit import Chem, RDLogger
        from rdkit.Chem.rdchem import BondType as BT
        from rdkit.Chem.rdchem import HybridizationType
        RDLogger.DisableLog('rdApp.*')

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        with open(self.raw_paths[1], 'r') as f:
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in f.read().split('\n')[1:-1]]
            y = torch.tensor(target, dtype=torch.float)
            y = torch.cat([y[:, 3:], y[:, :3]], dim=-1)
            y = y * conversion.view(1, -1)

        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False,
                                   sanitize=False)

        data_list = []
        # cls_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip:
                continue

            N = mol.GetNumAtoms()

            conf = mol.GetConformer()
            pos = conf.GetPositions()
            pos = torch.tensor(pos, dtype=torch.float)

            type_idx = []
            atomic_number = []
            aromatic = []
            sp = []
            sp2 = []
            sp3 = []
            num_hs = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])
                atomic_number.append(atom.GetAtomicNum())
                aromatic.append(1 if atom.GetIsAromatic() else 0)
                hybridization = atom.GetHybridization()
                sp.append(1 if hybridization == HybridizationType.SP else 0)
                sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
                sp3.append(1 if hybridization == HybridizationType.SP3 else 0)

            z = torch.tensor(atomic_number, dtype=torch.long)

            rows, cols, edge_types = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                rows += [start, end]
                cols += [end, start]
                edge_types += 2 * [bonds[bond.GetBondType()]]

            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)
            edge_attr = one_hot(edge_type, num_classes=len(bonds))

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_type = edge_type[perm]
            edge_attr = edge_attr[perm]

            row, col = edge_index
            hs = (z == 1).to(torch.float)
            num_hs = scatter(hs[row], col, dim_size=N, reduce='sum').tolist()

            x1 = one_hot(torch.tensor(type_idx), num_classes=len(types))
            x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, num_hs],
                              dtype=torch.float).t().contiguous()
            
            name = mol.GetProp('_Name')

            #MODIFICATION

            x3 = torch.zeros((len(mol.GetAtoms()), 768), dtype=torch.float16) #Create tensor of zeros, 768 is the size of the embedding dimension
            smiles = Chem.MolToSmiles(mol)
            
            #Remove all extra hydrogens
            cleaned_smiles = re.sub(r"(\[\(?H\)?])", "", smiles)
            cleaned_smiles = re.sub(r"\(\)", "", cleaned_smiles)

            # Get the embedding of the SMILES string from the language model
            tokenized_input = tokenize(tokenizer, cleaned_smiles, device)
            embedding = get_embedding(model, tokenized_input) 

            smiles_list = list(cleaned_smiles) # Split SMILES string along each character
            atom_order = [int(x) if x != '' else 0 for x in mol.GetProp('_smilesAtomOutputOrder').replace('[', '').replace(']', '').split(',')][:-1] 
            #Get the order in which the graph node atoms are inserted into the list 

            ### Create a mapping from each token in the smiles string to a node index, being None if it is a hydrogen 
            smiles_list_to_node_idx_len = len(smiles_list)
            smiles_list_to_node_idx = [None] * len(smiles_list) #Mapping from correspdoning token in smiles list to molecule graph node
            jdx = 0
            for idx in atom_order:
                atom_type = list(types.keys())[int(torch.argmax(x1[int(idx)]))]
                if atom_type != 'H': #If an H is in the atom output order it is implicit
                    while jdx < smiles_list_to_node_idx_len and atom_type != smiles_list[jdx].upper(): 
                        #If they are not the same, there is an extra hydrogen that should be skipped over 
                        #I'm still not quite sure why but rdkit adds hydrogens in something like [NH3+] or [C@@H], where the N and C are in the graph but the explicit Hs are not
                        jdx += 1

                    if jdx < smiles_list_to_node_idx_len:
                        smiles_list_to_node_idx[jdx] = idx
                        jdx += 1

            for smiles_idx, x_idx in enumerate(smiles_list_to_node_idx): 
                if x_idx != None: # We need to put in the embedding only when it is -1
                    x3[x_idx] = embedding[0][smiles_idx + 1] #We add one to avoid the [CLS] token
            
            # cls_list.append(embedding[0][0].to(torch.device("cpu")))
            # del embedding
            #MODIFICATION

            x = torch.cat([x1, x2, x3], dim=-1)

            data = Data(
                x=x,
                z=z,
                pos=pos,
                edge_index=edge_index,
                smiles=cleaned_smiles,
                edge_attr=edge_attr,
                y=y[i].unsqueeze(0),
                name=name,
                idx=i,
            )

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
            # break

        self.save(data_list, self.processed_paths[0])


if __name__ == "__main__":
    dataset = CustomQM9(root = data_dir)

    data = dataset[0]
    print(data)
    print(data.pos)




