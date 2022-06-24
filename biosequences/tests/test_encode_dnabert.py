'''Bla bla

'''
from biosequences.tokenizers import DNABertTokenizer
from biosequences.transformers import DNABertEncoder
from transformers import BertConfig, BertModel

VOCAB_FILE = './tests_data/vocabfile_3wordlength.txt'
HIDDEN_SIZE = 48
SEQ_TEST = 'AGGACGAACGCTGGCGGCGTGCTTAACACATGCAAGTCGAACGAGAGGACATGAAAAGCTTGCTTTTTATGAAATCTAGTGGCAAACGGGTGAGTAAC'

def test_simple_encode():
    tokenizer = DNABertTokenizer(vocab_file=VOCAB_FILE, word_length=3, stride=1, do_lower_case=True)
    config = BertConfig(vocab_size=tokenizer.vocab_size,
                        hidden_size=HIDDEN_SIZE,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=256)
    dnabert = DNABertEncoder(config=config)
    inputs = tokenizer(SEQ_TEST, return_tensors='pt')
    outputs = dnabert(**inputs)

    assert outputs['pooler_output'].shape[1] == HIDDEN_SIZE

def test_simple_encode2():
    tokenizer = DNABertTokenizer(vocab_file=VOCAB_FILE, word_length=3, stride=1, do_lower_case=True)
    config = BertConfig(vocab_size=tokenizer.vocab_size,
                        hidden_size=HIDDEN_SIZE,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=256)
    dnabert = BertModel(config=config)
    inputs = tokenizer(SEQ_TEST, return_tensors='pt')
    outputs = dnabert(**inputs)

    assert outputs['pooler_output'].shape[1] == HIDDEN_SIZE


if __name__ == '__main__':
    test_simple_encode()
    test_simple_encode2()