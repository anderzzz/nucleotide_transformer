'''Bla bla

'''
from biosequences import dna_nucleotide_alphabet, NucleotideVocabCreator
from biosequences import DNABertTokenizer

def test_foobar():
    dna_vocab = NucleotideVocabCreator(dna_nucleotide_alphabet, do_lower_case=True).generate(3)
    with open('./tmp_test.txt', 'w') as fout:
        dna_vocab.save(fout)
    tokenizer = DNABertTokenizer('./tmp_test.txt', word_length=3, stride=1, do_lower_case=True)
    print (tokenizer('AATGCGT'))

if __name__ == '__main__':
    test_foobar()
