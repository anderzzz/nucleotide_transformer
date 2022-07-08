'''Bla bla

'''
class Phrasifier(object):
    '''Bla bla

    '''
    def __init__(self, stride, word_length, do_upper_case=True, do_lower_case=False):
        self.stride = stride
        self.word_length = word_length
        if do_lower_case and do_upper_case:
            raise ValueError('The phrasifier cannot make both upper and lower case; reset parameters')
        self.upper = do_upper_case
        self.lower = do_lower_case

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
        elif self.lower:
            phrase = phrase.lower()

        return phrase
