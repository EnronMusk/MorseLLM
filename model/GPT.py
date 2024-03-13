import torch
from torch import nn

import os
import sys

#For safe imports of everything
notebook_directory = os.getcwd()
parent_directory = os.path.dirname(notebook_directory)
sys.path.insert(False, parent_directory)

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
max_batches = 50
eval_iterative = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eta = 1e-4
tempature = 1

class feedForward(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.ffwd = nn.Sequential(
            nn.Linear(hidden_size, feedforward_multiplier * hidden_size),
            nn.ReLU(),
            nn.Linear(feedforward_multiplier * hidden_size, hidden_size)
            )
        
    def forward(self, hidden_state):
        return self.ffwd(hidden_state)
    
class block(nn.Module):

    def __init__(self):
        super().__init__()

        self.mhead = multiHeadAttention()
        self.ffwd = feedForward()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, embedding):
        attn_embedding = self.mhead(embedding)
        attn_embedding = embedding + self.ln1(attn_embedding) #LayerNorm + Add

        new_embedding = self.ffwd(attn_embedding)
        new_embedding = embedding + self.ln2(new_embedding) #LayerNorm + Add
        return new_embedding

class attnHead(nn.Module):
    
    def __init__(self):
        super().__init__()  

        self.keys = nn.Linear(hidden_size, hidden_size / attn_heads, bias = False)
        self.queries = nn.Linear(hidden_size, hidden_size / attn_heads, bias = False)
        self.values = nn.Linear(hidden_size, hidden_size / attn_heads, bias = False)

    def forward(self, embeddings):

        keys = self.keys(embeddings) # B x T x head_size
        queries = self.queries(embeddings)
        values = self.values(embeddings)

        #attn scores
        dot_product = queries @ keys.transpose(-2, -1) # B x T x T

        triangular_matrix_ones = torch.tril(torch.ones(context_len, context_len))

        #Create masked attention
        default_weights = dot_product.masked_fill(triangular_matrix_ones == 0, float('-inf')) * (hidden_size / attn_heads)**-0.5

        #Assemble the attn matrix
        attn_matrix = torch.softmax(default_weights, dim=-1)
        new_embedding = attn_matrix @ values # B x T x T @ # B x T x head_size = B x T x head_size

        return new_embedding
    
class multiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.mhead = nn.ModuleList([attnHead() for _ in range(attn_heads)])
        self.proj_matrix = nn.Linear(hidden_size, hidden_size)

    def forward(self, embedding):

        m_attn_output = torch.concat([head.forward(embedding) for head in self.mhead], dim=1) #Concatenate along batch
        
        return self.proj_matrix(m_attn_output)

class transformerModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(context_len, hidden_size)

        self.ln = nn.LayerNorm(hidden_size)

        blocks_to_add = [block() for _ in range(n_layers)]
        self.blocks = nn.Sequential(*blocks_to_add) #Set up the layers here

        self.lm_head = nn.Linear(hidden_size, vocab_size)


    def forward(self, tokens):

        token_embeddings = self.embedding(tokens)
        position_embeddings = self.position_embedding(torch.arrange(context_len, device=device))
        total_embeddings = token_embeddings + position_embeddings
        
        final_embeddings = self.blocks(total_embeddings)
        final_embeddings = self.ln(final_embeddings)
        logits = self.lm_head(final_embeddings)

        return logits
    
    @torch.no_grad()
    def generate():
        return None
    
    def trainModel(self, train) -> None:

        model = self.to(device)
        params = list(model.parameters())
        optimizer = torch.optim.Adam(params=params, lr=eta)

        train_inputs = train['train_inputs']
        train_labels = train['train_targets']

        for i, batch in enumerate(list[zip(train_inputs, train_labels)]):

            #Evaluate and stop training settings
            if i % eval_iterative: self.evaluateModel(step=i, train=train)
            elif i > max_batches: break

            batch_inputs = batch[0].to(device)
            batch_labels = batch[1].to(device)

            logits, loss = self.computeLoss(inputs=batch_inputs, labels=batch_labels)

            #Training Core
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        self.evaluateModel()
    
    def computeLoss(self, inputs, labels) -> None:

        logits = self.forward(inputs) # B x T x C

        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits.view(batch_size * context_len, vocab_size), labels.view(batch_size * context_len))

        return logits, loss
    
    @torch.no_grad()
    def evaluateModel(self, step : int, train, test) -> None:

        self.eval()
        train_inputs = train['train_inputs']
        train_labels = train['train_targets']

        logits, loss = self.computeLoss(inputs=train_inputs, labels=train_labels)

        print("Step {step} Training Loss: {loss}")


        self.train()