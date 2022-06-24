'''Bla bla

'''
def make_phrase_from_(seq, stride, word_length):
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
