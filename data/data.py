
import sys
import os

#For safe imports of everything
notebook_directory = os.getcwd()
parent_directory = os.path.dirname(notebook_directory)
sys.path.insert(False, parent_directory)

from model.GPT import batch_size
from model.GPT import context_len

from tokenizer import tokenize

import math
import torch

from torch.utils.data import DataLoader

def createBatchTensor(train_pct : float,):
    """
    Creates a train and test tokenized dataset read for training and generation.
    """

    file = open('shakespear_encrypted.txt')
    content = file.read()
    file.close()

    tok_content = tokenize(content)
    n = len(tok_content)

    #Create tain and test datasets
    train = tok_content[math.floor(n * train_pct):]
    test = tok_content[:math.floor(n * train_pct)]

    def __assembleTensor(tokenized_data):

        iterate_index = [context_len * i for i in range(len(tokenized_data) // context_len)]
        inputs = [tokenized_data[i:i + context_len] for i in iterate_index]
        targets = [tokenized_data[i+1:i + context_len + 1] for i in iterate_index]

        batch_index = [batch_size * i for i in range(len(inputs) // batch_size - 1)]
        batched_inputs = [torch.tensor(inputs[i:i + batch_size]) for i in batch_index]
        batched_targets = [torch.tensor(targets[i+1:i + batch_size + 1]) for i in batch_index]

        batched_inputs = torch.stack(batched_inputs)
        batched_targets = torch.stack(batched_targets)

        return batched_inputs, batched_targets

    return __assembleTensor(train), __assembleTensor(test)


