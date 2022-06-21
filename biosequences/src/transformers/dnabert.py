'''Bla bla

'''
from torch import nn
from transformers import BertModel, BertConfig

from biosequences import DNABertTokenizer, DNABertTokenizerFast

class DNABertEncoder(nn.Module):
    '''Bla bla

    '''
    def __init__(self,
                 tokenizer,
                 config
                 ):
        super(DNABertEncoder, self).__init__()

        if (not isinstance(tokenizer, DNABertTokenizer)) or \
                (not isinstance(tokenizer, DNABertTokenizerFast)):
            raise TypeError('The tokenizer of invalid type')

        if not isinstance(config, BertConfig):
            raise TypeError('The configuration of invalid type')

        self.bert = BertModel(config=config)


