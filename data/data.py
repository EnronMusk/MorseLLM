
import sys
import os

#For safe imports of everything
notebook_directory = os.getcwd()
parent_directory = os.path.dirname(notebook_directory)
sys.path.insert(False, parent_directory)

from model.GPT import batch_size
from model.GPT import context_len

from data.tokenizer import tokenize

import math
import torch

def createBatchTensor(train_pct : float,):
    """
    Creates a train and test tokenized dataset read for training and generation.
    """

    file = open(parent_directory + r'\\data\\shakespear_encrypted.txt')
    content = file.read()
    file.close()

    tok_content = tokenize(content)
    n = len(tok_content)

    #Create tain and test datasets
    train = tok_content[math.floor(n * train_pct):]
    test = tok_content[:math.floor(n * train_pct)]

    def __assembleTensor(tokenized_data, type : str):

        iterate_index = [context_len * i for i in range(len(tokenized_data) // context_len)]
        inputs = [tokenized_data[i:i + context_len] for i in iterate_index]
        targets = [tokenized_data[i+1:i + context_len + 1] for i in iterate_index]

        batch_index = [batch_size * i for i in range(len(inputs) // batch_size - 1)]
        batched_inputs = [torch.tensor(inputs[i:i + batch_size]) for i in batch_index]
        batched_targets = [torch.tensor(targets[i:i + batch_size]) for i in batch_index]

        batched_inputs = torch.stack(batched_inputs)
        batched_targets = torch.stack(batched_targets)

        return {type + '_inputs': batched_inputs, type + '_targets': batched_targets} #Return a dictionary package

    return __assembleTensor(train, 'train'), __assembleTensor(test, 'test')


