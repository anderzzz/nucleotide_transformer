'''Bla bla

'''
from transformers import BertTokenizer, BertTokenizerFast

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

    def __call__(self, seq):
        if isinstance(seq, (list,tuple)):
            text = [self._make_phrase_from_(s) for s in seq]
        else:
            text = self._make_phrase_from_(seq)

        return super().__call__(text)

    def _make_phrase_from_(self, seq):
        if not isinstance(seq, str):
            raise TypeError('Encountered sequence that is not string: {}'.format(seq))

        lower_ind = len(seq) - self.word_length + 1
        if lower_ind < 1:
            raise RuntimeError('Sequence too short: {}'.format(seq))
        chunks = [seq[ind: ind + self.word_length] for ind in range(0, lower_ind, self.stride)]
        phrase = ' '.join(chunks)

        return phrase


class DNABertTokenizerFast(BertTokenizerFast):
    '''Bla bla

    '''
    def __init__(self):
        super(DNABertTokenizerFast, self).__init__(
            'dummy.txt'
        )

        pass