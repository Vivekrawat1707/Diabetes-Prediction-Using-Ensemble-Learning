import csv
from math import sqrt
from random import shuffle

def load_data(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            dataset.append([float(x) for x in row])
    return dataset

def normalize_data(dataset):
    min_max = []
    for i in range(len(dataset[0]) - 1):  # Exclude outcome column
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        min_max.append((value_min, value_max))
    
    normalized = []
    for row in dataset:
        norm_row = []
        for i in range(len(row) - 1):
            if min_max[i][1] - min_max[i][0] == 0:
                norm_row.append(0)
            else:
                norm_row.append((row[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0]))
        norm_row.append(row[-1])  # Add outcome
        normalized.append(norm_row)
    
    return normalized, min_max

def split_dataset(dataset, split_ratio=0.8):
    train_size = int(len(dataset) * split_ratio)
    dataset_copy = dataset.copy()
    shuffle(dataset_copy)
    return dataset_copy[:train_size], dataset_copy[train_size:]