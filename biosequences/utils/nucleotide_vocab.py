'''Nucleotide language generators

'''
from dataclasses import dataclass
from itertools import product, chain

@dataclass
class NucleotideAlphabet:
    letters: set
    letters_extended: set
    missing: str

    def is_standard_letter_(self, x):
        return x in self.letters

dna_nucleotide_alphabet = NucleotideAlphabet(letters={'A','T', 'C', 'G'},
                                           letters_extended=set(),
                                           missing='N'
                                           )
rna_nucleotide_alphabet = NucleotideAlphabet(letters={'A','U', 'C', 'G'},
                                           letters_extended=set(),
                                           missing='N'
                                           )

class NucleotideVocabCreator(object):
    '''Bla bla

    '''
    def __init__(self, alphabet,
                 do_lower_case=False,
                 do_upper_case=True,
                 extended_letters=False,
                 missing=False,
                 unk_token='[UNK]', sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]', mask_token='[MASK]'):

        self.vocab = None
        self.character_set = alphabet.letters
        if extended_letters:
            self.character_set += alphabet.letters_extended
        if missing:
            self.character_set += alphabet.missing

        if do_lower_case and do_upper_case:
            raise ValueError('Vocabulary set to be both all upper and all lower, not possible; reset arguments')
        if do_lower_case:
            self.character_set = [token.lower() for token in self.character_set]
        if do_upper_case:
            self.character_set = [token.upper() for token in self.character_set]

        self.special_tokens = [token for token in [unk_token, sep_token, pad_token, cls_token, mask_token] if not token is None]

    def print(self):
        print_str = ''
        for letters_ordered in self.vocab:
            print_str += '{}\n'.format(self._unify_word_(letters_ordered))
        return print_str

    def generate(self, word_length):
        self.vocab = product(sorted(self.character_set), repeat=word_length)
        self.vocab = chain(iter(self.special_tokens), self.vocab)
        return self

    def save(self, fout):
        for letters_ordered in self.vocab:
            fout.write('{}\n'.format(self._unify_word_(letters_ordered)))

    def _unify_word_(self, letters_ordered):
        return ''.join(letters_ordered)
