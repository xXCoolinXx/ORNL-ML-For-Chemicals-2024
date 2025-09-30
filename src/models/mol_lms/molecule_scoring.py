import rdkit
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import Crippen
import numpy as np
import math
import gzip
import pickle
import re
import scipy.stats
from scoring_interface import ScoringInterface

def smiles_to_mol(smiles):
    """Convert smiles to rdkit molecule.

    Args:
        smiles (str): smiles string representation of a molecule

    Returns:
        rdkit.Chem.rdchem.Mol

    """
    try:
        return Chem.MolFromSmiles(smiles, sanitize=True)
    except:
        return None

def mol_to_canonical_smiles(mol):
    """Convert rdkit molecule to canonical smiles string.

    Args:
        mol (rdkit.Chem.rdchem.Mol): rdkit molecule

    Returns:
        str representation of rdkit molecules (canonical)

    """
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def remap(x, x_min, x_max):
    """Translate and scale a given input.

    Args:
        x (float): original value
        x_min (float): value to subtract and lower bound for scale
        x_max (float): upper bound for scale

    Returns:
        float value translated and scaled

    """
    return (x - x_min) / (x_max - x_min)

class MoleculeScoring(ScoringInterface):
    """Class to score smiles sequences."""

    def __init__(self, scoring_names, selection_names, scoring_parameters=None, data_column_name='smiles', fitness_column_name='fitness', fitness_function=scipy.stats.hmean):
        """Constructor for MoleculeScoring class.

        Args:
            scoring_names (List[str]): List of names for scoring functions to use
            selection_names (List[str]): List of names for selection functions to use
            scoring_parameters (Dict[str, str]): Dictionary of parameters needed for scoring functions
            data_column_name (str): Name used data column
            fitness_column_name (str): Name used for fitness column
            fitness_function (function): Function used to calculate fitness score from selection metrics

        """
        super().__init__()

        # setup scoring parameters
        if scoring_parameters is None:
            scoring_parameters = {}

        # Dictionary storing mapping from names to scoring functions
        self._name_to_function = self.get_name_to_function_dict()

        # store variables
        self._data_column_name = data_column_name
        self._fitness_column_name = fitness_column_name
        self._fitness_function = fitness_function

        # exempt brackets for data cleaning
        self._exempt_brackets = {'[C@@H]', '[C@H]', '[C@]', '[C@@]', '[nH]'}

        # check that data and fitness column names are not part of possible scoring functions
        if self.data_column_name in self._name_to_function:
            raise ValueError('Error: data column name ' + self.data_column_name + ' cannot be ' + ', '.join(self._name_to_function.keys()))

        if self.fitness_column_name in self._name_to_function:
            raise ValueError('Error: data column name ' + self.data_column_name + ' cannot be ' + ', '.join(self._name_to_function.keys()))

        # setup for synth based on scoring_parameters
        self._sa_model = None
        if 'synth' in scoring_names:
            if 'sa_model' not in scoring_parameters:
                raise KeyError('Error: sa_model not in scoring_parameters, file location needed to use synth score')
            self._sa_model = {i[j]: float(i[0]) for i in pickle.load(gzip.open(scoring_parameters['sa_model'])) for j in range(1, len(i))}

        # store names of functions to be used in scoring - make sure that selection names are subset
        self._scoring_names = scoring_names
        for name in selection_names:
            if name not in self._scoring_names:
                self._scoring_names.append(name)

        # store selection names and check that they are subset of scoring names
        self._selection_names = selection_names

        # check whether scoring names are in self._name_to_function
        for name in scoring_names:
            if name not in self._name_to_function:
                raise KeyError('Error: ' + name + ' not an implemented scoring function. Options are ' + ', '.join(self._name_to_function.keys()))

    def get_name_to_function_dict(self):
        """Get dictionary that maps string to scoring functions.

        Note:
            To add scoring functions, inherit from MoleculeScoring and override

        Returns:
            Dict[str, function] to relate names to scoring functions

        """
        return {
            'synth': self._synthetic_accessibility_with_default,
            'drug': self._qed_with_default,
            'sol': self._crippen_mol_logp_with_default,
            'number': self._number_with_default
        }

    def generate_scores(self, mols):
        """Generate scores for list of rdkit molecules.

        Args:
            mols (List[rdkit.Chem.rdchem.Mol]): List of rdkit molecules

        Returns:
            Dict[str, List[float]] where keys are the scoring function names and the values are the scores

        """
        output = {}

        # apply scoring functions to molecules
        for scoring_name in self._scoring_names:
            output[scoring_name] = list(map(self._name_to_function[scoring_name], mols))
        
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

    def prepare_data_for_scoring(self, smiles):
        """Prepare smiles str for scoring.

        Args:
            smiles (str): smiles string representation of a molecule

        Returns:
            rdkit.Chem.rdchem.Mol if valid smiles or None if not valid smiles

        """
        # remove any spaces
        smiles = smiles.replace(' ','')

        # only allow brackets from exempt list
        all_brackets = re.findall(r'\[.*?\]', smiles)
        for bracket in all_brackets:
            if bracket not in self._exempt_brackets:
                return None

        # check for multiple molecules or wildcards
        if ('.' in smiles) or ('*' in smiles):
            return None

        # attempt conversion to molecule
        return smiles_to_mol(smiles)

    def make_canonical(self, mol):
        """Generate canonical smiles string for molecule.

        Args:
            mol (rdkit.Chem.rdchem.Mol): input molecule

        Returns: 
            str of canonical smiles (or None if not valid)

        """
        return mol_to_canonical_smiles(mol)

    @property
    def column_names(self):
        """Get list of names used in scoring dictionary.

        Returns:
            List[str] of names used in scoring dictionary

        """
        all_names = [self._data_column_name] + list(self._scoring_names) + [self._fitness_column_name]
        return all_names

    @property
    def selection_names(self):
        """Get list of names used for selection.

        Returns:
            List[str] of names using for selection (i.e. fitness) scoring

        """
        return self._selection_names

    @property
    def data_column_name(self):
        """Get name of data column.

        Returns:
            str with name of data column

        """
        return self._data_column_name

    @property
    def fitness_column_name(self):
        """Get name of fitness column.

        Returns:
            str with name of data column

        """
        return self._fitness_column_name

    def _qed_with_default(self, mol, default=0.0):
        """ Generate quantitative estimation of drug-likenss

        Args:
            mol (rdkit.Chem.rdchem.Mol): molecule to score
            default (float): Default value if scoring fails

        Returns:
            float score value

        """
        try:
            return QED.qed(mol)
        except:
            return default

    # octanol-water partition coefficient
    def _crippen_mol_logp_with_default(self, mol, default=-3.0, norm=True):
        """Generate Crippen MolLogP.

        Note:
            For remap values see https://github.com/nicola-decao/MolGAN/blob/master/utils/molecular_metrics.py

        Args:
            mol (rdkit.Chem.rdchem.Mol): molecule to score
            default (float): Default raw value if scoring fails
            norm (bool): Option to normalize output

        Returns:
            float score value

        """
        try:
            score = Crippen.MolLogP(mol)
        except:
            score = default
        if norm:
            score = np.clip(remap(score, -2.12178879609, 6.0429063424), 0.0, 1.0)
        return score

    # synthetic accesibility
    def _synthetic_accessibility_with_default(self, mol, default=10, norm=True):
        """Generate synthesizability score.

        Note:
            For remap values see https://github.com/nicola-decao/MolGAN/blob/master/utils/molecular_metrics.py

        Args:
            mol (rdkit.Chem.rdchem.Mol): molecule to score
            default (float): Default raw value if scoring fails
            norm (bool): Option to normalize output

        Returns:
            float score value

        """
        try:
            score = self._compute_sas(mol)
        except:
            score = default
        if norm:
            score = np.clip(remap(score, 5, 1.5), 0.0, 1.0)
        return score

    def _number_with_default(self, mol, default=0):
        """Generate number of atoms.

        Args:
            mol (rdkit.Chem.rdchem.Mol): molecule to score
            default (float): Default raw value if scoring fails

        Returns:
            float score value

        """
        try:
            score = mol.GetNumAtoms()
        except:
            score = default
        return score

    def _compute_sas(self, mol):
        """Helper function to compute synthesizability score.

        Note: 
            For more details see https://github.com/nicola-decao/MolGAN/blob/master/utils/molecular_metrics.py

        Args:
            mol (rdkit.Chem.rdchem.Mol): molecule to score

        Returns:
            float score value

        """
        fp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        score1 = 0.
        nf = 0
        # for bitId, v in fps.items():
        for bitId, v in fps.items():
            nf += v
            sfp = bitId
            score1 += self._sa_model.get(sfp, -4) * v

        # check added to prevent divide by zero
        if len(fps) > 0:
            score1 /= nf

        # features score
        nAtoms = mol.GetNumAtoms()
        nChiralCenters = len(Chem.FindMolChiralCenters(
            mol, includeUnassigned=True))
        ri = mol.GetRingInfo()
        nSpiro = Chem.rdMolDescriptors.CalcNumSpiroAtoms(mol)
        nBridgeheads = Chem.rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        nMacrocycles = 0
        for x in ri.AtomRings():
            if len(x) > 8:
                nMacrocycles += 1

        sizePenalty = nAtoms ** 1.005 - nAtoms
        stereoPenalty = math.log10(nChiralCenters + 1)
        spiroPenalty = math.log10(nSpiro + 1)
        bridgePenalty = math.log10(nBridgeheads + 1)
        macrocyclePenalty = 0.

        # ---------------------------------------
        # This differs from the paper, which defines:
        #  macrocyclePenalty = math.log10(nMacrocycles+1)
        # This form generates better results when 2 or more macrocycles are present
        if nMacrocycles > 0:
            macrocyclePenalty = math.log10(2)

        score2 = 0. - sizePenalty - stereoPenalty - \
                    spiroPenalty - bridgePenalty - macrocyclePenalty

        # correction for the fingerprint density
        # not in the original publication, added in version 1.1
        # to make highly symmetrical molecules easier to synthetise
        score3 = 0.
        if nAtoms > len(fps):
            score3 = math.log(float(nAtoms) / len(fps)) * .5

        sascore = score1 + score2 + score3

        # need to transform "raw" value into scale between 1 and 10
        min = -4.0
        max = 2.5
        sascore = 11. - (sascore - min + 1) / (max - min) * 9.
        
        
        # smooth the 10-end
        
        # previously part of code - this doesn't make sense
        # if a score is near 8 (8.0000001) it will be set to a very low score
        # if sascore > 8.:
        #     sascore = 8. + math.log(sascore + 1. - 9.)

        if sascore > 10.:
            sascore = 10.0
        elif sascore < 1.:
            sascore = 1.0

        return sascore

# some sample use cases
if __name__ == '__main__':
    print('Example of using molecule scoring\n', flush=True)

    # construct MoleculeScoring object
    metrics = ['drug', 'sol', 'number']
    selection_metrics = ['drug']
    metric_parameters = {}
    molecule_scoring = MoleculeScoring(metrics, selection_metrics, metric_parameters)

    # example smiles    
    smiles_examples = ['c1ccccc1', 'OCc1ccccc1', 'Brc1ccccc1C2CCCC2', 'junk']

    # prepare data for scoring
    molecule_examples = []
    for s in smiles_examples:
        mol = molecule_scoring.prepare_data_for_scoring(s)
        if mol is not None:
            molecule_examples.append(mol)

    # score data
    scores = molecule_scoring.generate_scores(molecule_examples)

    # number of samples and column names
    number_of_samples = len(scores[molecule_scoring.data_column_name])
    column_names = molecule_scoring.column_names

    # print column names
    for key in column_names:
        print(key, end='\t')
    print()

    # print data
    for i in range(number_of_samples):
        for key in column_names:
            print(scores[key][i], end='\t')
        print()



