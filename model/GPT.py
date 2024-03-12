import torch
from torch import nn

#Global hyper-paramaters

#attention
hidden_size = 32
attn_heads = 4

#feedforward
feedforward_multiplier = 4

#layers
n_layers = 6
vocab_size = 3

#other
batch_size = 4
context_len = 8
max_batches = 5000
eval_iterative = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eta = 1e-4
tempature = 1



class feedForward(nn.Module):
    
    def __init__(self, hidden_size):
        super().__init__()

        self.fwd = nn.Sequential(
            nn.Linear(hidden_size, feedforward_multiplier * hidden_size),
            nn.ReLU(),
            nn.Linear(feedforward_multiplier * hidden_size, hidden_size)
            )

class attnHead(nn.Module):
    
    def __init__(self):
        super().__init__()  

class gptModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(context_len, hidden_size)

        self.lm_head = nn.Linear(hidden_size, vocab_size)


    def forward(self, idx, targets=None):
        return None
    
    def generate():
        return None
    
    def trainModel():
        return None
    
    def computeLoss():
        return None