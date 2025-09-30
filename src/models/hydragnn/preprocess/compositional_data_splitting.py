##############################################################################
# Copyright (c) 2021, Oak Ridge National Laboratory                          #
# All rights reserved.                                                       #
#                                                                            #
# This file is part of HydraGNN and is distributed under a BSD 3-clause      #
# license. For the licensing terms see the LICENSE file in the top-level     #
# directory.                                                                 #
#                                                                            #
# SPDX-License-Identifier: BSD-3-Clause                                      #
##############################################################################

import math
import collections
import torch
import sklearn

# function to return key for any value
def get_keys(dictionary, val):
    keys = []
    for key, value in dictionary.items():
        if val == value:
            keys.append(key)
    return keys


def get_max_graph_size(dataset):
    max_graph_size = int(0)
    for data in dataset:
        max_graph_size = max(max_graph_size, data.num_nodes)
    return max_graph_size


def get_elements_list(dataset):
    ## Identify all the elements present in at least one configuration
    elements_list = torch.FloatTensor()

    for data in dataset:
        elements = torch.unique(data.x[:, 0])
        elements_list = torch.cat((elements_list, elements), 0)

    elements_list = torch.unique(elements_list)

    return elements_list


def create_dictionary_from_elements_list(elements_list: list):
    dictionary = {}

    for index, element in enumerate(elements_list):
        dictionary[element.item()] = index

    return dictionary


def create_dataset_categories(dataset):
    max_graph_size = get_max_graph_size(dataset)
    power_ten = math.ceil(math.log10(max_graph_size))
    elements_list = get_elements_list(dataset)
    elements_dictionary = create_dictionary_from_elements_list(elements_list)

    dataset_categories = []

    for data in dataset:
        elements, frequencies = torch.unique(data.x[:, 0], return_counts=True)
        category = 0
        for element, frequency in zip(elements, frequencies):
            category += frequency.item() * (
                10 ** (power_ten * elements_dictionary[element.item()])
            )
        dataset_categories.append(category)

    return dataset_categories


def duplicate_unique_data_samples(dataset, dataset_categories):
    counter = collections.Counter(dataset_categories)
    keys = get_keys(counter, 1)
    augmented_data = []
    augmented_data_category = []
    augmented_data_indicies = [] # Indices to the ORIGINAL dataset
    dataset_indices = list(range(len(dataset)))

    for idx, (data, category) in enumerate(zip(dataset, dataset_categories)):
        augmented_data_indicies.append(idx)
        if category in keys:
            # Data augmentation on unique elements to allow additional splitting
            augmented_data.append(data.clone())
            augmented_data_category.append(category)

    if isinstance(dataset, list):
        dataset.extend(augmented_data)
    else:
        dataset.dataset.extend(augmented_data)
    dataset_indices.extend(augmented_data_indicies)
    dataset_categories.extend(augmented_data_category)

    return dataset, dataset_categories, dataset_indices


def generate_partition(
    sss: sklearn.model_selection.StratifiedShuffleSplit, dataset, dataset_categories
):
    parition1_indices = []
    partition2_indices = []
    partition1_set = []
    partition2_set = []

    for train_index, val_test_index in sss.split(dataset, dataset_categories):
        parition1_indices = train_index.tolist()
        partition2_indices = val_test_index.tolist()

    for index in parition1_indices:
        partition1_set.append(dataset[index])

    for index in partition2_indices:
        partition2_set.append(dataset[index])

    return partition1_set, partition2_set, parition1_indices, partition2_indices


def compositional_stratified_splitting(dataset, perc_train):
    """Given the dataset and the percentage of data you want to extract from it, method will
    apply stratified sampling where X is the dataset and Y is are the category values for each datapoint.
    In the case each structure contains 2 types of atoms, the category will
    be constructed as such: number of atoms of type 1 + number of atoms of type 2 * 100.
    Parameters
    ----------
    dataset: [Data]
        A list of Data objects representing a structure that has atoms.
    subsample_percentage: float
        Percentage of the dataset.
    Returns
    ----------
    [Data]
        Subsample of the original dataset constructed using stratified sampling.
    """
    dataset_categories = create_dataset_categories(dataset)
    dataset, dataset_categories, augmented_to_original_indices = duplicate_unique_data_samples(
        dataset, dataset_categories
    )

    sss_train = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, train_size=perc_train, random_state=0
    )
    trainset, val_test_set, train_indices, val_test_indices = generate_partition(sss_train, dataset, dataset_categories)

    train_indices = [augmented_to_original_indices[i] for i in train_indices] 
    val_test_indices = [augmented_to_original_indices[i] for i in val_test_indices]

    val_test_dataset_categories = create_dataset_categories(val_test_set)
    val_test_set, val_test_dataset_categories, augmented_to_original_indices_val_test = duplicate_unique_data_samples(
        val_test_set, val_test_dataset_categories
    )

    sss_valtest = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, train_size=0.5, random_state=0
    )
    valset, testset, val_indices, test_indices = generate_partition(
        sss_valtest, val_test_set, val_test_dataset_categories
    )
    print(min(val_test_indices), max(val_test_indices))
    print(len(val_test_indices))
    print(max(val_indices))
    val_indices = [val_test_indices[augmented_to_original_indices_val_test[val_indices[i]]] for i in range(len(val_indices))]
    test_indices = [val_test_indices[augmented_to_original_indices_val_test[test_indices[i]]] for i in range(len(test_indices))]
    
    print(min(train_indices), max(train_indices))
    print(min(val_indices), max(val_indices))
    print(min(test_indices), max(test_indices))

    return trainset, valset, testset, train_indices, val_indices, test_indices
