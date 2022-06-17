'''Bla bla

'''
from biosequences import create_bert_objs

TEXT_INP = 'This time Kubrick has truly outdone himself. In two hours and nineteen minutes he ' + \
           'tells the story of the dawn of mankind to the birth of a new species. Brilliant special effect. ' + \
           'Arguably the most sympathetic character is a computer, voiced by the great Douglas Rain.'

model, tokenizer = create_bert_objs()
inputs = tokenizer(TEXT_INP, return_tensors='pt')
output = model(inputs)
print (output)