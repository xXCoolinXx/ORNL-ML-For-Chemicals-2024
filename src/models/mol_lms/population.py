from molecule_scoring import MoleculeScoring
import numpy as np
import rdkit
import rdkit.Chem

def randomize_smiles(input_smiles, retries=5):
    """Generate a randmized version of an input smiles
    
    Args:
        input_smiles (str): smiles string representation of a molecule

    Returns:
        str with a randomized smiles for the input molecule
    """
    for _ in range(retries):
        try:
            mol = rdkit.Chem.MolFromSmiles(input_smiles)
            atom_order = list(range(mol.GetNumAtoms()))
            np.random.shuffle(atom_order)
            new_mol = rdkit.Chem.RenumberAtoms(mol, atom_order)
            randomized_smiles = rdkit.Chem.MolToSmiles(new_mol, canonical=False)
            if randomized_smiles != input_smiles:
                return randomized_smiles
        except:
            continue

    return None

class Population():
    """Class to store a population of sequences and apply mutation/recombination/selection."""

    def __init__(self, gan_operators, scoring_operator):
        """Contructor for Population class.

        Args:
            gan_operators (List[gan]): List of gans used for generating mutations
            scoring_operator (ScoringInterface): Scoring operator to score population sequences

        """
        super().__init__()

        # store evoluation_operators and scoring_operator
        self._gan_operators = gan_operators
        self._scoring_operator = scoring_operator

        # intialize population dict
        self._population_dict = {}
        self._data_column_name = self._scoring_operator.data_column_name
        self._fitness_column_name = self._scoring_operator.fitness_column_name
        self._column_names = self._scoring_operator.column_names

    @property
    def population_dict(self):
        """Get population dictionary.

        Returns:
            Dict[str,] with data and scores for population

        """
        return self._population_dict

    @property
    def population_size(self):
        """ Get population size.

        Returns;
            int with population size
        """
        if len(self._population_dict) > 0:
            return len(self._population_dict[self._data_column_name])
        else:
            return 0

    @property
    def population_sequences(self):
        """Get population sequences.
        
        Returns:
            List[str] with values for data column name key in population dict

        """
        return self._population_dict[self._data_column_name]

    def read_population_dict_from_file(self, population_file, population_size=1000, delimiter='\t'):
        """Read a population file and store contents in population_dict

        Args:
            population_file (str): Path to population file (should be a delimited file with header row with column names)
            population_size (int): Number of data rows in the population.
            delimiter (str): Delimiter for parsing population file

        """
        # create a dictionary to store column names and order from file
        header_dict = {}
        reverse_header_dict = {}

        # clear population_dict
        self._population_dict = {}

        # store sequences that must be scored
        sequences_to_score = []

        # flag to generate metrics if not provided
        generate_metrics = False

        with open(population_file, 'r') as input_file:
            row_counter = -1
            for row in input_file:
                row_counter += 1
                # read header row
                if row_counter == 0:
                    header_names = [x.strip() for x in row.split(delimiter)]
                    column_counter = 0
                    for name in header_names:
                        header_dict[name] = column_counter
                        column_counter += 1

                    # construct reverse map
                    for key in header_dict:
                        reverse_header_dict[header_dict[key]] = key

                    # make sure that file has data column
                    if self._data_column_name not in header_dict:
                        raise ValueError('Error: %s does not have %s as a header column' % (population_file, self._data_column_name))
                
                    # check that all metrics are provided
                    required_column_names = self._scoring_operator.column_names
                    for column_name in required_column_names:
                        self._population_dict[column_name] = []
                        if column_name not in header_dict:
                            generate_metrics = True

                    continue

                # non-header rows
                row_split = [x.strip() for x in row.split(delimiter)]

                if not generate_metrics:
                    for i in range(len(row_split)):
                        if reverse_header_dict[i] != self._data_column_name:
                            self._population_dict[reverse_header_dict[i]].append(float(row_split[i]))
                        else:
                            self._population_dict[reverse_header_dict[i]].append(row_split[i])
                else:
                    sequences_to_score.append(row_split[header_dict[self._data_column_name]])

                # stop reading if population_size is met
                if population_size >= 0 and row_counter >= population_size:
                    break

        # check if scoring is needed
        if len(sequences_to_score) > 0:
            self._population_dict = self.sequences_to_population_dict(sequences_to_score)

        # fill population by making random copies to desired size
        if len(self._population_dict[self._data_column_name]) < population_size:
            self._fill_population_dict(population_size)

    def generate_child_population_dict(self, mutation_samples, weighted=False, previous_set=None, db_dict=None, batch_size=10, return_valid=False, add_randomized_smiles=False):
        """Generate a child population dict from current population dict

        Args:
            mutation_samples (List[int]): Number of mutation samples for each gan object
            weight (bool): Option to weight sampling for mutation
            previous_set (set[str]): Set of previously visited sequences
            db_dict (Dict[str,]): Dictionary with keys cursor and query_string
            batch_size (int): Batch size for generating mutations and recombinations
            return_valid (bool): Option to return number of valid molecules generated

        Returns:
            Dict[str,] with child population dict

        """
        # check that mutation samples are valid
        if len(mutation_samples) != len(self._gan_operators):
            raise ValueError('Error length of mutation samples is not equal to length of evolution operators')

        # setup defaults
        if previous_set is None:
            previous_set = set()

        if db_dict is None:
            db_dict = {}

        # store generated sequences from mutations
        possible_sequences = []
        masked_sequences = []

        # mutation samples
        samples = None
        for i in range(len(mutation_samples)):
            samples = self._sample_population_dict(mutation_samples[i], weighted)
            for j in range(0, mutation_samples[i], batch_size):
                start_index = j
                end_index = min(start_index + batch_size, mutation_samples[i])
                total_batch = samples[start_index:end_index].tolist()

                # add randomized smiles if requested
                if add_randomized_smiles:
                    for example in samples[start_index:end_index]:
                        r_smiles = randomize_smiles(example)
                        if r_smiles is not None:
                            total_batch.append(r_smiles)      

                generated_sequences, m_sequences = self._gan_operators[i].evaluate_generator(total_batch)
                possible_sequences += generated_sequences
                masked_sequences += m_sequences

        child_population_dict = None
        child_valid = -1
        if return_valid:
            child_population_dict, child_valid = self.sequences_to_population_dict(possible_sequences, previous_set, db_dict, return_valid)
            # if (child_valid/len(possible_sequences)) < 0.2:
            #     print('Example Generated Sequences')
            #     for i in range(25):
            #         if i % 5 == 0:
            #             print('Masked: ', masked_sequences[i//5])
            #         print(possible_sequences[i], flush=True)
            return child_population_dict, child_valid
        else:
            child_population_dict = self.sequences_to_population_dict(possible_sequences, previous_set, db_dict, return_valid)
            return child_population_dict

        # # generate dictionary from possible sequences
        # return self.sequences_to_population_dict(possible_sequences, previous_set, db_dict, return_valid)


    def merge_child_population_dict(self, child_population_dict, max_size=-1):
        """Merge child population dict with current population dict.

        Args:
            child_population_dict (Dict[str,]): Dictionary for child population
            max_size (int): allows population to grow to max_size, otherwise merged population maintains previous size

        Returns:
            int number of child population merged into original population

        """
        # save population size before merge
        original_population_size = self.population_size

        # append to current population
        for key in self._column_names:
            self._population_dict[key] += child_population_dict[key]

        # allow population growth if max_size is set
        cutoff_size = original_population_size
        if max_size > 0:
            cutoff_size = min(max_size, len(self._population_dict[self._data_column_name]))

        # population size is maintained by the merge
        selection_index = None
        if len(self._scoring_operator.selection_names) < 1:
            selection_index = np.random.choice(self.population_size, cutoff_size, replace=False)
        else:
            selection_index = np.argsort(-1.0*np.array(self._population_dict[self._fitness_column_name]))[:cutoff_size]

        # apply selection
        for key in self._column_names:
            self._population_dict[key] = [self._population_dict[key][x] for x in selection_index]

        # count children accepted
        children_accepted = np.sum(selection_index >= original_population_size)

        return children_accepted

    def write_population_dict_header(self, output_file, add_epoch=False):
        """Write header for population dict

        Args:
            output_file (file object): Write enabled file object

        """
        if not add_epoch:
            output_file.write('\t'.join(self._column_names) + '\n')
        else:
            output_file.write('\t'.join(self._column_names + ['epoch']) + '\n')

    def write_population_dict_values(self, output_file, population_dict=None, epoch=None):
        """Write values for population dict

        Args:
            output_file (file object): Write enabled file object
            population_dict (Dict[str,]): Dictionary used for output

        """
        # default is to write current population
        population_dict_to_write = self._population_dict if population_dict is None else population_dict
        population_size = len(population_dict_to_write[self._data_column_name])
        for i in range(population_size):
            row_data = []
            for key in self._column_names:
                if key == self._data_column_name:
                    row_data.append(population_dict_to_write[key][i])
                else:
                    row_data.append('%.6f' % (population_dict_to_write[key][i]))

            # option to write epoch to output file
            if epoch is not None:
                row_data.append(str(epoch))

            output_file.write('\t'.join(row_data) + '\n')

    def get_population_averages(self):
        """Get average of scoring metrics for population
        
        Returns:
            Dict[str, float] with averages for population metrics

        """
        averages_dict = {}
        for key in self._column_names:
            if key != self._data_column_name:
                averages_dict[key] = np.mean(self._population_dict[key])

        return averages_dict

    def _sample_population_dict(self, number_of_samples, weighted):
        """Return sample from data column of population dict

        Args:
            number_of_samples (int): Number of samples to draw
            weighted (bool): Option to weight samples by softmax of fitness

        Returns:
            List[str] with sampled sequences from data column of population dict

        """
        if weighted:
            # softmax weights from fitness
            weights = np.exp(self._population_dict[self._fitness_column_name])
            weights /= np.sum(weights)
            return np.random.choice(self._population_dict[self._data_column_name], number_of_samples, p=weights)
        else:
            return np.random.choice(self._population_dict[self._data_column_name], number_of_samples)

    def sequences_to_population_dict(self, sequences, previous_set=None, db_dict=None, return_valid=False):
        """Generation a population_dict from a list of sequences

        Args:
            sequences (List[str]): List of sequences for population
            previous_set (set[str]): Set of previously visited sequences
            db_dict (Dict[str,]): Dictionary with keys cursor and query_string
            return_valid (bool): Option to return count of valid molecules

        Returns:
            Dict[str,] produced by the scoring operator

        """
        # setup defaults
        if previous_set is None:
            previous_set = set()

        if db_dict is None:
            db_dict = {}

        sequences_to_keep = []

        # valid sequences produced
        valid_counter = 0

        for sequence in sequences:

            # check if sequence is viable
            prepared_data = self._scoring_operator.prepare_data_for_scoring(sequence)
            if prepared_data is not None:

                valid_counter += 1

                # attempt to make canonical - for cases like molecules generation where cleaned_data and sequence don't have same type
                canonical_data = self._scoring_operator.make_canonical(prepared_data)

                # check if data has already been recorded
                if canonical_data in previous_set:
                    continue

                # check if data has already been recorded in db
                if ('cursor' in db_dict) and ('query_string' in db_dict):
                    canonical_query = (canonical_data,)
                    cursor = db_dict['cursor']
                    cursor.execute(db_dict['query_string'], canonical_query)
                    if cursor.fetchone()[0] == 1:
                        continue

                # add to population
                sequences_to_keep.append(prepared_data)
                previous_set.add(canonical_data)

        # generate population
        if return_valid:
            return self._scoring_operator.generate_scores(sequences_to_keep), valid_counter
        else:
            return self._scoring_operator.generate_scores(sequences_to_keep)

    def train_gans(self, train_loader, train_flags=None):
        """Train GANs associated with the population.
        
        Args:
            train_loader (torch.utils.data.DataLoader): ataloader to iterate through dataset

        Returns:
            two str with comma separated discriminator and generator loss
        
        """
        train_disc_loss = []
        train_gen_loss = []
        counter = 0
        for gan in self._gan_operators:
            flag = True
            if train_flags is not None:
                flag = (train_flags[counter] == 1)

            if flag:
                _, disc_loss, gen_loss = gan.train_epoch(train_loader)
                train_disc_loss.append(disc_loss)
                train_gen_loss.append(gen_loss)
            else:
                train_disc_loss.append(0.0)
                train_gen_loss.append(0.0)

            counter += 1

        return ','.join(['%0.4f' % x for x in train_disc_loss]), ','.join(['%0.4f' % x for x in train_gen_loss])

    def _fill_population_dict(self, desired_size):
        """Fill a population_dict to a desired size by making random copies.

        Args:
            desired_size (int): Desired number of elements for each key in the population_dict

        """
        current_size = len(self._population_dict[self._data_column_name])
        if current_size < desired_size:
            copy_indices = np.random.choice(current_size, desired_size - current_size)
            for index in copy_indices:
                for key in self._population_dict:
                    self._population_dict[key].append(self._population_dict[key][index])

# some sample use cases
if __name__ == '__main__':
    import sys

    print('Example of using Population\n', flush=True)

    arguments = sys.argv
    if len(arguments) < 2:
        print('Error: not enough arguments provided - python population.py smiles_file')
        sys.exit(1)

    # location of input data
    smiles_input_file = arguments[1]

    # turn off logging
    from rdkit import RDLogger
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.CRITICAL)

    # construct MoleculeScoring object
    metrics = ['drug', 'sol', 'number']
    selection_metrics = ['drug']
    metric_parameters = {}
    molecule_scoring = MoleculeScoring(metrics, selection_metrics, metric_parameters)

    # construct Population
    population = Population(molecule_scoring)
    print('Population size after construction:', population.population_size)

    # read population data
    population.read_population_dict_from_file(smiles_input_file)
    print('Population size after reading:', population.population_size)

    # population averages
    print('Population averages', end='\t')
    population_averages = population.get_population_averages()
    for key in population_averages:
        print('%s: %.4f' % (key, population_averages[key]), end='\t')
    print()
