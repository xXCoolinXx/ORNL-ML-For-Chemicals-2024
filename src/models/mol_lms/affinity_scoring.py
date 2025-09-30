from math import sqrt
from molecule_scoring import MoleculeScoring
from molecule_scoring import mol_to_canonical_smiles
from transformers import BertConfig, BertTokenizer, AutoTokenizer
import json
import torch
import re
import scipy
import numpy as np
from token_splits import pretokenizer_dict

######################################################################################
# Scoring Class
######################################################################################

class AffinityScoring(MoleculeScoring):

    def __init__(self, scoring_names, selection_names, scoring_parameters=None, data_column_name='smiles', fitness_column_name='fitness', fitness_function=scipy.stats.hmean):
        super().__init__(scoring_names, selection_names, scoring_parameters=scoring_parameters, data_column_name=data_column_name, fitness_column_name=fitness_column_name, fitness_function=fitness_function)

        # checks on parameters for affinity
        if "affinity" in scoring_names:
            from sentence_transformers import SentenceTransformer
            if "fine_tuned_model_names" not in scoring_parameters:
                raise KeyError('Error: fine_tuned_model_names not in scoring parameters, needed to use affinity')           
            if "protein_sequence" not in scoring_parameters:
                raise KeyError('Error: protein_sequence not in scoring parameters, needed to use affinity')
            if "device" not in scoring_parameters:
                raise KeyError('Error: device not in scoring parameters, needed to use affinity')

            self._protein_sequence = scoring_parameters['protein_sequence']

            self.fine_tuned_models = []
            for model_name in scoring_parameters['fine_tuned_model_names']:
                model = SentenceTransformer(model_name, device=scoring_parameters['device'])
                model.eval()
                self.fine_tuned_models.append(model)

    def _affinity(self, mols):
        model_input = [{'protein': [self._protein_sequence], 'ligand': [mol_to_canonical_smiles(x)]} for x in mols]
        outputs = []
        with torch.no_grad():
            for model in self.fine_tuned_models:
                outputs.append(model.encode(model_input).squeeze(-1).squeeze(-1))
        mean_values = np.mean(outputs,axis=0)
        mean_values = mean_values / 10.0
        mean_values = np.clip(mean_values, 0.0, 1.0)

        return mean_values.tolist()

    def get_name_to_function_dict(self):
        output_dict = super().get_name_to_function_dict()
        output_dict['affinity'] = self._affinity
        return output_dict

    def generate_scores(self, mols):
        output = {}

        # apply scoring functions to molecules
        for scoring_name in self._scoring_names:
            if (scoring_name == "affinity"):
                continue
            output[scoring_name] = list(map(self._name_to_function[scoring_name], mols))

        batch_size = len(mols) 
        if ("affinity" in self._scoring_names):
            output["affinity"] = []

            for i in range(0, len(mols), batch_size):
                start_index = i
                end_index = min(start_index + batch_size, len(mols))
                mean_values = self._affinity(mols[start_index:end_index])
                output["affinity"] += mean_values

        # always return data as well
        output[self.data_column_name] = list(map(mol_to_canonical_smiles, mols))

        # return fitness
        output[self._fitness_column_name] = []
        for i in range(len(output[self.data_column_name])):
            fitness_array = [output[x][i] for x in self._selection_names]
            if len(fitness_array) > 0:
                output[self._fitness_column_name].append(self._fitness_function(fitness_array))
            else:
                output[self._fitness_column_name].append(-1.0)

        return output
