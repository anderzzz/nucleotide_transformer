'''Bla bla

'''
from biosequences.scripts.train_maskedlm import fit_bert_maskedlm

TEST_DATA = './tests_data/'

fit_bert_maskedlm(
     folder_seq_raw=TEST_DATA, seq_raw_format='csv', seq_raw_file_pattern='test_xaa.csv', upper_lower='upper',
     folder_seq_sentence=TEST_DATA, seq_sentence_prefix='test_maskedlm_',
     vocab_file='vocab.txt', create_vocab=True, word_length_vocab=3, stride=1,
     chunk_size=5000,
     masking_probability=0.15,
     bert_config_kwargs={},
     folder_training_output=TEST_DATA
)