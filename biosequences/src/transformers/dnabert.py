'''Bla bla

'''
from torch import nn
from transformers import BertModel, BertConfig

from biosequences import DNABertTokenizer, DNABertTokenizerFast

class DNABertEncoder(BertModel):
    '''Bla bla

    '''
    def __init__(self,
                 config,
                 add_pooling_layer=True
                 ):
        super(DNABertEncoder, self).__init__(config=config,
                                             add_pooling_layer=add_pooling_layer)

    def forward(self, *args, **kwargs):
        bert_code = super().forward(*args, **kwargs)
        print (bert_code)
        raise RuntimeError('Charmed')

