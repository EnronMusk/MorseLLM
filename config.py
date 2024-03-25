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
        self.chars = ['\n', ' ', '!', '"', '&', "'", '(', ')', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '>', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|', '}']
        self.vocab_size = len(self.chars)
        self.directory = r'\\data\\shakespeardata.txt'
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