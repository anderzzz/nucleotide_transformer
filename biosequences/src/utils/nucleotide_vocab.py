'''Nucleotide language generators

'''
from dataclasses import dataclass
from itertools import permutations

@dataclass
class NucleotideAlphabet:
    letters: set
    letters_extended: set
    missing: str

    def is_standard_letter_(self, x):
        return x in self.letters

DNANucleotideAlphabet = NucleotideAlphabet(letters={'A','T', 'C', 'G'},
                                           letters_extended=set(),
                                           missing='X'
                                           )
RNANucleotideAlphabet = NucleotideAlphabet(letters={'A','U', 'C', 'G'},
                                           letters_extended=set(),
                                           missing='X'
                                           )

class NucleotideVocabCreator(object):
    '''Bla bla

    '''
    def __init__(self, alphabet,
                 extended_letters=False,
                 missing=False):

        self.vocab = None
        self.character_set = alphabet.letters
        if extended_letters:
            self.character_set += alphabet.letters_extended
        if missing:
            self.character_set += alphabet.missing

    def print(self):
        print_str = ''
        for letters_ordered in self.vocab:
            print_str += '{}\n'.format(self._unify_word_(letters_ordered))
        return print_str

    def generate(self, word_length):
        self.vocab = permutations(self.character_set, word_length)
        return self

    def save(self, fout):
        for letters_ordered in self.vocab:
            fout.write('{}\n'.format(self._unify_word_(letters_ordered)))

    def _unify_word_(self, letters_ordered):
        return ''.join(letters_ordered)