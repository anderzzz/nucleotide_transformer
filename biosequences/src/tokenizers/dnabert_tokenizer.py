'''Bla bla

'''
from transformers import BertTokenizer, BertTokenizerFast

class DNABertTokenizer(BertTokenizer):
    '''Bla bla

    '''
    def __init__(self):
        super(DNABertTokenizer, self).__init__(
            'dummy.txt'
        )

        pass

class DNABertTokenizerFast(BertTokenizerFast):
    '''Bla bla

    '''
    def __init__(self):
        super(DNABertTokenizerFast, self).__init__(
            'dummy.txt'
        )

        pass