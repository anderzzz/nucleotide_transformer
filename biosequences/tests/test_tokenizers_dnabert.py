'''Bla bla

'''
from biosequences import dna_nucleotide_alphabet, NucleotideVocabCreator
from biosequences import DNABertTokenizer, DNABertTokenizerFast

SEQ_STR = 'AATGCGT'
IDS_SEQ_STR = [3,8,19,62,43,32,1]
SEQ_BATCH = ['AATGCGT', 'GGGGT']
IDS_SEQ_BATCH = [[3,8,19,62,43,32,1], [3,47,47,48,1]]

def test_tokenize_str():
    dna_vocab = NucleotideVocabCreator(dna_nucleotide_alphabet, do_lower_case=True).generate(3)
    with open('./tmp_test.txt', 'w') as fout:
        dna_vocab.save(fout)
    tokenizer = DNABertTokenizer('./tmp_test.txt', word_length=3, stride=1, do_lower_case=True)
    out = tokenizer(SEQ_STR)
    assert out.input_ids == IDS_SEQ_STR

def test_tokenize_batch():
    dna_vocab = NucleotideVocabCreator(dna_nucleotide_alphabet, do_lower_case=True).generate(3)
    with open('./tmp_test.txt', 'w') as fout:
        dna_vocab.save(fout)
    tokenizer = DNABertTokenizer('./tmp_test.txt', word_length=3, stride=1, do_lower_case=True)
    out = tokenizer(SEQ_BATCH)
    assert out.input_ids == IDS_SEQ_BATCH

def test_tokenize_fast_str():
    dna_vocab = NucleotideVocabCreator(dna_nucleotide_alphabet, do_lower_case=True).generate(3)
    with open('./tmp_test.txt', 'w') as fout:
        dna_vocab.save(fout)
    tokenizer = DNABertTokenizerFast('./tmp_test.txt', word_length=3, stride=1, do_lower_case=True)
    assert tokenizer.is_fast

    out = tokenizer(SEQ_STR)
    assert out.input_ids == IDS_SEQ_STR

def test_tokenize_fast_batch():
    dna_vocab = NucleotideVocabCreator(dna_nucleotide_alphabet, do_lower_case=True).generate(3)
    with open('./tmp_test.txt', 'w') as fout:
        dna_vocab.save(fout)
    tokenizer = DNABertTokenizerFast('./tmp_test.txt', word_length=3, stride=1, do_lower_case=True)
    assert tokenizer.is_fast

    out = tokenizer(SEQ_BATCH)
    assert out.input_ids == IDS_SEQ_BATCH


if __name__ == '__main__':
    test_tokenize_str()
    test_tokenize_batch()
    test_tokenize_fast_str()
    test_tokenize_fast_batch()
