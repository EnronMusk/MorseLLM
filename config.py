class modelConfig():
    def __init__(self):

        #attention
        self.hidden_size = 384
        self.attn_heads = 6
        self.head_size = self.hidden_size // self.attn_heads 

        #feedforward
        self.feedforward_multiplier = 4

        #layers
        self.n_layers = 6

        #other
        self.batch_size = 64
        self.context_len = 192
        self.max_batches = 5000
        self.num_epoch = 15

        self.eta = 2e-4
        self.tempature = 1

        #data related
        self.chars = ['(', ':', "'", '_', 'H', 'L', 'I', '1', 'W', 'K', '3', '.', '8', 'G', '9', 'R', ';', '4', 'J', '?', '0', 'P', 'U', '&', '5', '7', 'V', '-', ' ', 'Q', 'O', 'Y', ')', 'D', 'Z', '\n', '"', 'E', '2', 'B', 'A', 'T', '6', 'F', 'C', 'X', 'S', '!', ',', 'M', 'N']
        self.vocab_size = len(self.chars)
        self.directory = r'\\data\\shakespear_upper.txt'
        self.total_tokens = None #Found in the tokenizer. Used to justify the context length increase.

class modelConfigMorse():
    def __init__(self):

        #attention
        self.hidden_size = 384
        self.attn_heads = 6
        self.head_size = self.hidden_size // self.attn_heads 

        #feedforward
        self.feedforward_multiplier = 4

        #layers
        self.n_layers = 6

        #other
        self.batch_size = 64
        self.context_len = 650 #192 * ~3.4 since roughly 3.37x more tokens per character.
        self.max_batches = 5000
        self.num_epoch = 15

        self.eta = 2e-4
        self.tempature = 1

        #vocab and tokens
        self.chars = [' ', '.', '-', '/']
        self.vocab_size = len(self.chars)
        self.directory = r'\\data\\shakespear_encrypted.txt'
        self.total_tokens = None #Found in the tokenizer. Used to justify the context length increase.