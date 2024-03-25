import torch

def tokenize(string : str, config) -> list[int]:
    chars = config.chars

    encode_mapping = { ch:i for i,ch in enumerate(chars)}
    encode = lambda string : [encode_mapping[c] for c in string]

    return encode(string)

def detokenize(tensor : torch.Tensor, config) -> str:
    chars = config.chars

    decode_mapping = { i:ch for i,ch in enumerate(chars)}
    decode = lambda tensor : ''.join([decode_mapping[tok] for tok in tensor])

    return decode(tensor)
