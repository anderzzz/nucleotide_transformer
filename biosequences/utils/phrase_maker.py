'''Bla bla

'''
class Phrasifier(object):
    '''Bla bla

    '''
    def __init__(self, stride, word_length, upper=False):
        self.stride = stride
        self.word_length = word_length
        self.upper = upper

    def __call__(self, seq):
        if not isinstance(seq, str):
            raise TypeError('Encountered sequence that is not string: {}'.format(seq))

        lower_ind = len(seq) - self.word_length + 1
        if lower_ind < 1:
            raise RuntimeError('Sequence too short: {}'.format(seq))
        chunks = [seq[ind: ind + self.word_length] for ind in range(0, lower_ind, self.stride)]
        phrase = ' '.join(chunks)
        if self.upper:
            phrase = phrase.upper()

        return phrase
