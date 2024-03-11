chars = " .-"

def tokenize(string : str) -> list[int]:

    encode_mapping = { ch:i for i,ch in enumerate(chars)}
    encode = lambda string : [encode_mapping[c] for c in string]

    return encode(string)

def detokenize(string : str) -> list[int]:

    decode_mapping = { i:ch for i,ch in enumerate(chars)}
    decode = lambda string : ''.join([decode_mapping[c] for c in string])

    return decode(string)
