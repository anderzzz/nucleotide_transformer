'''Bla bla

'''
import torch

from transformers import BertTokenizer, BertTokenizerFast

from biosequences.utils import dna_nucleotide_alphabet, NucleotideVocabCreator, Phrasifier

SEQ_STR = 'AATGCGT'
IDS_SEQ_STR = [3,8,19,62,43,32,1]
SEQ_BATCH = ['AATGCGT', 'GGGGT']
IDS_SEQ_BATCH = [[3,8,19,62,43,32,1], [3,47,47,48,1]]

def test_tokenize_str():
    dna_vocab = NucleotideVocabCreator(dna_nucleotide_alphabet, do_lower_case=True).generate(3)
    with open('tmp_test.txt', 'w') as fout:
        dna_vocab.save(fout)
    phrasifier = Phrasifier(stride=1, word_length=3)
    tokenizer = BertTokenizer(vocab_file='tmp_test.txt', tokenize_chinese_chars=False)
    out = tokenizer(phrasifier(SEQ_STR))
    assert out.input_ids == IDS_SEQ_STR

def test_tokenize_str_ptreturn():
    dna_vocab = NucleotideVocabCreator(dna_nucleotide_alphabet, do_lower_case=True).generate(3)
    with open('tmp_test.txt', 'w') as fout:
        dna_vocab.save(fout)
    phrasifier = Phrasifier(stride=1, word_length=3)
    tokenizer = BertTokenizer(vocab_file='tmp_test.txt', tokenize_chinese_chars=False)
    out = tokenizer(phrasifier(SEQ_STR), return_tensors='pt')
    assert torch.equal(out.input_ids, torch.tensor(IDS_SEQ_STR).reshape(1,-1))

def test_tokenize_batch():
    dna_vocab = NucleotideVocabCreator(dna_nucleotide_alphabet, do_lower_case=True).generate(3)
    with open('tmp_test.txt', 'w') as fout:
        dna_vocab.save(fout)
    phrasifier = Phrasifier(stride=1, word_length=3)
    tokenizer = BertTokenizer('tmp_test.txt', tokenize_chinese_chars=False, do_lower_case=True)
    out = tokenizer([phrasifier(s) for s in SEQ_BATCH])
    assert out.input_ids == IDS_SEQ_BATCH

def test_tokenize_fast_str():
    dna_vocab = NucleotideVocabCreator(dna_nucleotide_alphabet, do_lower_case=True).generate(3)
    with open('tmp_test.txt', 'w') as fout:
        dna_vocab.save(fout)
    phrasifier = Phrasifier(stride=1, word_length=3)
    tokenizer = BertTokenizerFast('tmp_test.txt', do_lower_case=True, tokenize_chinese_chars=False)
    assert tokenizer.is_fast

    out = tokenizer(phrasifier(SEQ_STR))
    assert out.input_ids == IDS_SEQ_STR

def test_tokenize_fast_batch():
    dna_vocab = NucleotideVocabCreator(dna_nucleotide_alphabet, do_lower_case=True).generate(3)
    with open('tmp_test.txt', 'w') as fout:
        dna_vocab.save(fout)
    phrasifier = Phrasifier(stride=1, word_length=3)
    tokenizer = BertTokenizerFast('tmp_test.txt', word_length=3, do_lower_case=True)
    assert tokenizer.is_fast

    out = tokenizer([phrasifier(s) for s in SEQ_BATCH])
    assert out.input_ids == IDS_SEQ_BATCH

if __name__ == '__main__':
    test_tokenize_str()
    test_tokenize_str_ptreturn()
    test_tokenize_batch()
    test_tokenize_fast_str()
    test_tokenize_fast_batch()
