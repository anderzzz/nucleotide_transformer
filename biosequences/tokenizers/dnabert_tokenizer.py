'''Bla bla

'''
from transformers import BertTokenizer, BertTokenizerFast
from biosequences.utils import make_phrase_from_

class DNABertTokenizer(BertTokenizer):
    '''Bla bla

    '''
    def __init__(self, vocab_file,
                 stride=1, word_length=3,
                 do_lower_case=True, do_basic_tokenize=True,
                 unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]'):
        super(DNABertTokenizer, self).__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            tokenize_chinese_chars=False,
            strip_accents=False,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token
        )
        self.stride = stride
        self.word_length = word_length

    def __call__(self, seq, **kwargs):
        if isinstance(seq, (list,tuple)):
            text = [make_phrase_from_(s, self.stride, self.word_length) for s in seq]
        else:
            text = make_phrase_from_(seq, self.stride, self.word_length)

        return super().__call__(text, **kwargs)



class DNABertTokenizerFast(BertTokenizerFast):
    '''Bla bla

    '''
    def __init__(self, vocab_file,
                 stride=1, word_length=3,
                 do_lower_case=True, do_basic_tokenize=True,
                 unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]'):
        super(DNABertTokenizerFast, self).__init__(
            vocab_file=vocab_file,
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            tokenize_chinese_chars=False,
            strip_accents=False,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token
        )
        self.stride = stride
        self.word_length = word_length

    def __call__(self, seq, **kwargs):
        if isinstance(seq, (list,tuple)):
            text = [make_phrase_from_(s, self.stride, self.word_length) for s in seq]
        else:
            text = make_phrase_from_(seq, self.stride, self.word_length)

        return super().__call__(text, **kwargs)