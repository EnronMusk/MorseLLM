import torch
from torch import nn

import os
import sys
from typing import Tuple


#For safe imports of everything
notebook_directory = os.getcwd()
parent_directory = os.path.dirname(notebook_directory)
sys.path.insert(False, parent_directory)

from utils.plots import createDualLossPlot, createDualTrainPlot, createDualAccuracyPlot


device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

class feedForward(nn.Module):
    
    def __init__(self, hidden_size, feedforward_multiplier):
        super().__init__()

        self.ffwd = nn.Sequential(
            nn.Linear(hidden_size, feedforward_multiplier * hidden_size),
            nn.ReLU(),
            nn.Linear(feedforward_multiplier * hidden_size, hidden_size)
            )
        
    def forward(self, hidden_state) -> torch.Tensor:
        return self.ffwd(hidden_state)
    
class block(nn.Module):

    def __init__(self, hidden_size, attn_heads, head_size, feedforward_multiplier):
        super().__init__()

        self.mhead = multiHeadAttention(hidden_size, attn_heads, head_size)
        self.ffwd = feedForward(hidden_size, feedforward_multiplier)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, embedding) -> torch.Tensor:
        attn_embedding = embedding + self.mhead(self.ln1((embedding))) #LayerNorm + Add

        new_embedding = attn_embedding + self.ffwd(self.ln2(attn_embedding)) #LayerNorm + Add
        return new_embedding

class attnHead(nn.Module):
    
    def __init__(self, hidden_size, head_size):
        super().__init__()  

        self.keys = nn.Linear(hidden_size, head_size, bias = False)
        self.queries = nn.Linear(hidden_size, head_size, bias = False)
        self.values = nn.Linear(hidden_size, head_size, bias = False)


    def forward(self, embeddings) -> torch.Tensor:

        B, T, V = embeddings.shape #Grab the embeddings dimensions so we can perform self attention on dynamic sized inputs.

        keys = self.keys(embeddings) # B x T x head_size
        queries = self.queries(embeddings)
        values = self.values(embeddings)

        #attn scores
        dot_product = ((queries @ keys.transpose(-2, -1)) * (keys.shape[-1])**-0.5) # B x T x T
        #print(dot_product.shape)

        triangular_matrix_ones = torch.tril(torch.ones(T, T, device=device))

        #Create masked attention
        default_weights = (dot_product.masked_fill(triangular_matrix_ones == 0, float('-inf')))
        #print("weights")
        #print(default_weights.shape)

        #Assemble the attn matrix
        attn_matrix = torch.softmax(default_weights, dim=-1)
        new_embedding = attn_matrix @ values # B x T x T @ # B x T x head_size = B x T x head_size

        #print("new emb")
        #print(new_embedding.shape)

        return new_embedding
    
class multiHeadAttention(nn.Module):

    def __init__(self, hidden_size, attn_heads, head_size):
        super().__init__()
        self.mhead = nn.ModuleList([attnHead(hidden_size, head_size) for _ in range(attn_heads)])
        self.proj_matrix = nn.Linear(hidden_size, hidden_size, bias = False)

    def forward(self, embedding) -> torch.Tensor:

        m_attn_output = torch.cat([head.forward(embedding) for head in self.mhead], dim=-1) #Concatenate along batch
        #print("final output")
        #print(m_attn_output.shape)
        
        return self.proj_matrix(m_attn_output)

class transformerModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.context_len, config.hidden_size)

        self.ln = nn.LayerNorm(config.hidden_size)

        blocks_to_add = [block(config.hidden_size, config.attn_heads, config.head_size, config.feedforward_multiplier) for _ in range(config.n_layers)]
        self.blocks = nn.Sequential(*blocks_to_add) #Set up the layers here

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.apply(self.initWeights) #initialize weights near zero

    def initWeights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, tokens) -> torch.Tensor:
        batch_size, context_len = tokens.shape

        token_embeddings = self.embedding(tokens)
        position_embeddings = self.position_embedding(torch.arange(context_len, device=device)) 
        #print(token_embeddings.shape)
        #print(position_embeddings.shape)
        total_embeddings = token_embeddings + position_embeddings
        
        #print(total_embeddings.shape)
        final_embeddings = self.blocks(total_embeddings)
        final_embeddings = self.ln(final_embeddings)
        logits = self.lm_head(final_embeddings)

        return logits
    
    @torch.no_grad()
    def generate(self, tokens, max_new_tokens) -> torch.Tensor:

        for _ in range(max_new_tokens):

            tokens_cropped = tokens[:, -self.config.context_len:] #Only consider tokens up to the context length
            logits = self.forward(tokens_cropped)
            #print("logits")
            #print(logits)

            logits = logits[:, -1, :] # B x T x V ---> B x V
            #print(logits)

            probs = torch.softmax(logits, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
            #new_token = logits.argmax(-1).unsqueeze(0) #Choose the most likely token.

            #print(tokens)
            #print(new_token)

            tokens = torch.cat((tokens, new_token), dim=1)
            #print(tokens)
        return tokens
    
    def getParams(self) -> int:
        return sum([p.numel() for p in self.parameters()])
    
    def trainModel(self, train, test, plot=True) -> None:

        """" Trains the model and plots results. """

        self.to(device)

        config = self.config
        params = self.parameters()
        optimizer = torch.optim.Adam(params=params, lr=config.eta)

        train_inputs = train['train_inputs']
        train_labels = train['train_targets']

        #Initialize some arrays to keep track of model performance over time
        batch_train_losses, epoch_train_losses, epoch_test_losses, batch_train_acc, epoch_train_acc, epoch_test_acc, = [], [], [], [], [], []

        print("\u2714 Starting training...")
        for j in range(1, config.num_epoch+1):
            for i, batch in enumerate(list(zip(train_inputs, train_labels))):
                batch_inputs = batch[0].to(device)
                batch_labels = batch[1].to(device)

                #Evaluate and stop training settings
                #if i > config.max_batches: break

                logits, loss, acc = self.computeLoss(inputs=batch_inputs, labels=batch_labels)
                #Training Core
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_train_losses.append(loss.item())
                batch_train_acc.append(acc)

            #Evaluate model every epoch and track results
            train_loss, test_loss, train_acc, test_acc = self.evaluateModel(epoch=j, train=train, test=test)
            epoch_test_losses.append(test_loss)
            epoch_train_losses.append(train_loss)

            epoch_test_acc.append(test_acc)
            epoch_train_acc.append(train_acc)

        #Plot model performance over time
        if plot: 
            createDualTrainPlot(batch_train_losses, batch_train_acc)
            createDualLossPlot(epoch_train_losses, epoch_test_losses)
            createDualAccuracyPlot(epoch_train_acc, epoch_test_acc)
            
    def computeLoss(self, inputs, labels) -> Tuple[torch.Tensor, torch.Tensor, float]:

        """ returns the logits, loss and the accuracy respectively. """

        #labels are B x T
        #inputs are B x T x C

        config = self.config
        logits = self.forward(inputs) # B x T x C

        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(logits.view(config.batch_size * config.context_len, config.vocab_size), labels.view(config.batch_size * config.context_len))

        pred = logits.argmax(dim=-1) # B x T x C ---> B x T
        correct_tokens = (pred == labels).sum().item() # B x T == B x T
        total_tokens = labels.numel()

        return logits, loss, correct_tokens/total_tokens
    
    #utility method for evaluating a dataset by type
    def __evaluateData(self, inputs, outputs) -> Tuple[float, float]:

        """ Helper function to evalute the model. Evaluates a dataset of batches.
        \n returns mean loss and accuracy respectively"""

        losses = 0
        accuracy = 0
        for i,batch in enumerate(list(zip(inputs, outputs))):
                batch_inputs = batch[0].to(device)
                batch_labels = batch[1].to(device)

                logits, loss, acc = self.computeLoss(inputs=batch_inputs, labels=batch_labels)
                losses += loss.item()
                accuracy += acc #is a float
        
        return round(losses / i, 4), round(accuracy / i, 4) #Return the means.

    @torch.no_grad()
    def evaluateModel(self, epoch : int, train, test) -> Tuple[float, float]:

        """ evaluates the models losses and accuracy for both train and test.
            \n returns train/test losses and train/test acurracy respectively."""

        self.eval()

        train_inputs = train['train_inputs']
        train_labels = train['train_targets']

        test_inputs = test['test_inputs']
        test_labels = test['test_targets']

        train_mean, train_acc = self.__evaluateData(train_inputs, train_labels)
        test_mean, test_acc = self.__evaluateData(test_inputs, test_labels)

        #Print status update of losses + accuracy
        print(f"Epoch: {epoch} Training Loss: {train_mean} Test Loss: {test_mean} Train Accuracy {train_acc} Test Accuracy {test_acc}")

        self.train()

        return train_mean, test_mean, train_acc, test_acc