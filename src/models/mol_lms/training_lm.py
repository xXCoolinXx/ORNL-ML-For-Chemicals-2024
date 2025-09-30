import transformers
import torch
import numpy as np
import json
import rdkit.Chem
from rdkit import RDLogger
import argparse
from gan import Gan
from affinity_scoring import AffinityScoring
from population import Population
from datetime import datetime
import sys

############################################################################
# Parameters
############################################################################

parser = argparse.ArgumentParser()

# Input/Output
parser.add_argument('--config', type=str, default='./config/default.json', help='json config for scoring operators and gans')
parser.add_argument('--data_file', type=str, default='./data/example_smiles.tsv', help='file with smiles data in first column')
parser.add_argument('--output_directory', type=str, default=None, help='output directory for population and new sequences')
parser.add_argument('--run_id', type=str, default='run', help='run id used as a prefix in output files')

# Environment
parser.add_argument('--use_mpi', action='store_true', default=False, help='option to use mpi to determine device')
parser.add_argument('--data_file_postfix', action='store_true', default=False, help='option to use different postfix based on rank for data file')
parser.add_argument('--data_file_tag', type=str, default=None, help='tag for data file to be used with data_file_postfix option')

# Hyperparameters
parser.add_argument('--population_size', type=int, default=1000, help='maximum number of training samples to use')
parser.add_argument('--mutation_samples', type=str, default='1000', help='number of samples to be sent to the generator for evaluation')
parser.add_argument('--mutation_parameter', type=str, default='0.15', help='determines fraction of tokens that are masked for generator')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for model training')
parser.add_argument('--top_k', type=int, default=5, help='number of top predictions for generator evaluation')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.00005, help='learning rate for model training')
parser.add_argument('--generator_only', action='store_true', default=False, help='option to turn off GAN training')
parser.add_argument('--no_merge', action='store_true', default=False, help='option to turn off child population merge')
parser.add_argument('--add_randomized_smiles', action='store_true', default=False, help='option to add randomized smiles for generator evaluation')
parser.add_argument('--random_init', action='store_true', default=False, help='option to use random weights to intialize language model')
parser.add_argument('--reset_all_smiles', action='store_true', default=False, help='option to count novel based only on current population')
parser.add_argument('--mlm_loss', action='store_true', default=False, help='option to use mlm loss to train generator only')
parser.add_argument('--train_flags', type=str, default=None, help='comman separated list of 0 or 1 to determine if gan is trained')

args = parser.parse_args()

# run id
data_file = args.data_file
run_id = args.run_id
device = None
log_file = None
if args.use_mpi:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    device = 'cuda:%d' % (rank % 6)
    run_id = '%s_%d' % (args.run_id, rank)
    log_file_name = '%s/%s.log' % (args.output_directory, run_id)
    log_file = open(log_file_name, 'w')
    
    data_file_tag = ''
    if args.data_file_tag is not None:
        data_file_tag = '_' + args.data_file_tag

    if args.data_file_postfix:

        data_file = '%s_%d%s.tsv' % (data_file, rank, data_file_tag)

print(args, file=log_file)

# turn off rdkit logging - otherwise tons of output during metrics calculations
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

# read config for run
with open(args.config, 'r') as f:
    config = json.load(f)

# format for multiple generator model
mutation_samples_list = [int(x.strip()) for x in args.mutation_samples.split(',')]
mutation_parameter_list = [float(x.strip()) for x in args.mutation_parameter.split(',')]

# train flags
train_flags = None
if args.train_flags is not None:
    train_flags = [int(x) for x in args.train_flags.split(',')]

############################################################################
# Scoring
############################################################################

scoring_config = config['scoring_operator']
if device is not None:
    if 'device' in scoring_config['scoring_parameters']:
        scoring_config['scoring_parameters']['device'] = device
scoring_operator = AffinityScoring(**scoring_config)

############################################################################
# GAN Operators
############################################################################

gan_operators = []
gan_counter = 0
for operator_config in config['gan_operators']:
    if device is not None:
        if 'device' in operator_config:
            operator_config['device'] = device

    gan_operators.append(
        Gan(**operator_config,
            mutation_parameter=mutation_parameter_list[gan_counter], 
            lr=args.lr, 
            generator_only=args.generator_only, 
            top_k=args.top_k, 
            random_init=args.random_init)
            )
    gan_counter += 1

if train_flags is not None:
    if len(train_flags) != gan_counter:
        print('train_flags must be same length as number of gans', file=log_file, flush=True)
        sys.exit(1)

############################################################################
# Population
############################################################################

population = Population(gan_operators, scoring_operator)
population.read_population_dict_from_file(data_file, args.population_size)
all_smiles = set(population.population_sequences)
print('Data Samples: %d' % len(all_smiles), population.population_size, file=log_file, flush=True)

# output files
new_seq_output_file = None
population_output_file = None
if args.output_directory is not None:
    new_seq_output_file = open(args.output_directory + '/' + run_id + '_new_sequences.tsv', 'w')
    population.write_population_dict_header(new_seq_output_file, add_epoch=True)
    population_output_file = open(args.output_directory + '/' + run_id + '_population.tsv', 'w')
    population.write_population_dict_header(population_output_file)

############################################################################
# Model Training and Evaluation
############################################################################

# print intial conditions
# write to std out - time and novel
print('[%d/%d] time: %s\tvalid: %d\tnovel: %d\tpositive: %d\taccepted: %d\tdloss: %0.4f\tgloss: %0.4f'
    % (0, args.epochs,
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    0,
    0,
    0,
    0,
    0,
    0), file=log_file, flush=True, end='\t')

# write to std out - population averages
population_averages = population.get_population_averages()
for key in population_averages:
    print('%s: %.4f' % (key, population_averages[key]), file=log_file, end='\t')
print(file=log_file, flush=True)

# run training and selection
for i in range(args.epochs):

    # setup data loader with current population
    train_loader = torch.utils.data.DataLoader(population.population_sequences,batch_size=args.batch_size,shuffle=True)

    # train GAN
    train_disc_loss = '0.0000'
    train_gen_loss = '0.0000'
    if (not args.generator_only) or (args.mlm_loss):
        train_disc_loss, train_gen_loss = population.train_gans(train_loader, train_flags)

    # Generate child population using generator
    child_population_dict, children_valid = population.generate_child_population_dict(mutation_samples_list, previous_set=all_smiles, return_valid=True)
    children_novel = len(child_population_dict[scoring_operator.data_column_name])

    # eliminate children with zero fitness
    children_fitness = child_population_dict[population._fitness_column_name]
    non_zero_indices = np.nonzero(children_fitness)[0]
    children_positive = children_novel
    if len(non_zero_indices) < len(children_fitness):
        for key in child_population_dict:
            child_population_dict[key] = [child_population_dict[key][x] for x in non_zero_indices]
        children_positive = len(non_zero_indices)

    # merge population
    children_accepted = 0
    if not args.no_merge:
        children_accepted = population.merge_child_population_dict(child_population_dict)

    # option for all smiles
    if args.reset_all_smiles:
        all_smiles = set(population.population_sequences)

    # updated population averages
    population_averages = population.get_population_averages()

    # write to std out - time and novel
    print('[%d/%d] time: %s\tvalid: %d\tnovel: %d\tpositive: %d\taccepted: %d\tdloss: %s\tgloss: %s'
        % (i+1, args.epochs,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        children_valid,
        children_novel,
        children_positive,
        children_accepted,
        train_disc_loss,
        train_gen_loss), file=log_file, flush=True, end='\t')

    # write to std out - population averages
    for key in population_averages:
        print('%s: %.4f' % (key, population_averages[key]), file=log_file, end='\t')
    print(file=log_file, flush=True)

    # write new sequences
    if new_seq_output_file is not None:
        population.write_population_dict_values(new_seq_output_file, child_population_dict, epoch=i+1)

##########################################################################################
# Clean-up
##########################################################################################

# close total output file
if new_seq_output_file is not None:
    new_seq_output_file.close()

# write population to file
if population_output_file is not None:
    population.write_population_dict_values(population_output_file)
    population_output_file.close()

if log_file is not None:
    log_file.close()
