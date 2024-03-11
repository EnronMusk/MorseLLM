
from .model.GPT import batch_size

def createBatchTensor():

    file = open('shakespear_encrypted.txt')
    content = file.read()
    file.close()

