'''Bla bla

'''
from transformers import BertConfig, BertModel, BertTokenizer

from biosequences.utils import Phrasifier

VOCAB_FILE = 'tests_data/vocabfile_3wordlength.txt'
HIDDEN_SIZE = 48
SEQ_TEST = 'AGGACGAACGCTGGCGGCGTGCTTAACACATGCAAGTCGAACGAGAGGACATGAAAAGCTTGCTTTTTATGAAATCTAGTGGCAAACGGGTGAGTAAC'

def test_simple_encode():
    phrasifier = Phrasifier(stride=1, word_length=3)
    tokenizer = BertTokenizer(vocab_file=VOCAB_FILE, do_lower_case=True)
    config = BertConfig(vocab_size=tokenizer.vocab_size,
                        hidden_size=HIDDEN_SIZE,
                        num_hidden_layers=12,
                        num_attention_heads=12,
                        intermediate_size=256)
    bert = BertModel(config=config)
    inputs = tokenizer(phrasifier(SEQ_TEST), return_tensors='pt')
    outputs = bert(**inputs)

    assert outputs['pooler_output'].shape[1] == HIDDEN_SIZE

if __name__ == '__main__':
    test_simple_encode()