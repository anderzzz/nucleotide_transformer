'''WARNING WARNING

I THINK THESE CHILD CLASSES ARE UNNECESSARY AND SHOULD BE REPLACED BY STRAIGHTUP HUGGINFACE TRANSFORMERS

'''
from torch import nn
from transformers import BertModel, BertForSequenceClassification

from biosequences.tokenizers import DNABertTokenizer, DNABertTokenizerFast

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
        return bert_code


class DNABertSequenceClassifier(BertForSequenceClassification):
    '''Bla bla

    '''
    def __init__(self,
                 config,
                 ):
        super(DNABertSequenceClassifier, self).__init__(config=config)

    def forward(self, *args, **kwargs):
        label = super().forward(*args, **kwargs)
