import os
import sys

#For safe imports of everything
notebook_directory = os.getcwd()
parent_directory = os.path.dirname(notebook_directory)
sys.path.insert(False, parent_directory)

file = open(parent_directory + r'\\data\\shakespeardata.txt')
content = file.read()
file.close()

chars = sorted(list(set(content)))

# chars = " .-"

def tokenize(string : str) -> list[int]:

    encode_mapping = { ch:i for i,ch in enumerate(chars)}
    encode = lambda string : [encode_mapping[c] for c in string]

    return encode(string)

def detokenize(string : str) -> list[int]:

    decode_mapping = { i:ch for i,ch in enumerate(chars)}
    decode = lambda string : ''.join([decode_mapping[c] for c in string])

    return decode(string)
