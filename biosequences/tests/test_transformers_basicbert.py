'''Bla bla

'''
import torch
from transformers import BertConfig, BertTokenizer

from biosequences import BertTextSentimentClassifier

TEXT_INP = 'This time Kubrick has truly outdone himself. In two hours and nineteen minutes he ' + \
           'tells the story of the dawn of mankind to the birth of a new species. Brilliant special effect. ' + \
           'Arguably the most sympathetic character is a computer, voiced by the great Douglas Rain.'

DUMMY_INP = 'a1 a2 a3 a1 a2 a2 a3'
DUMMY_VOCAB = './tests_data/vocab_file_1.txt'
CUSTOM_IDS = [[3, 5, 6, 7, 5, 6, 6, 7, 1]]

def test_default_pretrained():
    model = BertTextSentimentClassifier(pretrained='bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(TEXT_INP, return_tensors='pt')
    output = model(inputs)
    print (output)

def test_custom_tokenizer():
    config = BertConfig(num_hidden_layers=6, num_attention_heads=6)
    model = BertTextSentimentClassifier(config=config)
    tokenizer = BertTokenizer(vocab_file=DUMMY_VOCAB)

    inputs = tokenizer(DUMMY_INP, return_tensors='pt')
    assert torch.equal(inputs.input_ids, torch.tensor(CUSTOM_IDS))

    output = model(inputs)
    print (output)

if __name__ == '__main__':
    test_default_pretrained()
    test_custom_tokenizer()