'''Bla bla

'''
from transformers import BertTokenizer, BertTokenizerFast

def _make_phrase_from_(seq, stride, word_length):
    '''Bla bla

    '''
    if not isinstance(seq, str):
        raise TypeError('Encountered sequence that is not string: {}'.format(seq))

    lower_ind = len(seq) - word_length + 1
    if lower_ind < 1:
        raise RuntimeError('Sequence too short: {}'.format(seq))
    chunks = [seq[ind: ind + word_length] for ind in range(0, lower_ind, stride)]
    phrase = ' '.join(chunks)

    return phrase

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

    def __call__(self, seq, return_tensors=None):
        if isinstance(seq, (list,tuple)):
            text = [_make_phrase_from_(s, self.stride, self.word_length) for s in seq]
        else:
            text = _make_phrase_from_(seq, self.stride, self.word_length)

        return super().__call__(text, return_tensors=return_tensors)


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

    def __call__(self, seq):
        if isinstance(seq, (list,tuple)):
            text = [_make_phrase_from_(s, self.stride, self.word_length) for s in seq]
        else:
            text = _make_phrase_from_(seq, self.stride, self.word_length)

        return super().__call__(text)