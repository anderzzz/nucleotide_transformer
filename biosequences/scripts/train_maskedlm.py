'''Bla bla

'''
from biosequences import NucleotideSequenceDataset
from biosequences import DNABertTokenizer, NucleotideVocabCreator, dna_nucleotide_alphabet

from transformers import BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer

def main(gb_data_folder=None,
         vocab_file='./vocab.txt', create_vocab=True, word_length_vocab=3,
         masking_probability=0.15):

    #
    # Construct Dataset
    if gb_data_folder is None:
        raise ValueError('The folder with Genbank data files must be provided in `gb_data_folder`')
    dataset = NucleotideSequenceDataset(gb_data_folder)

    #
    # Construct the tokenizer
    if create_vocab:
        dna_vocab = NucleotideVocabCreator(dna_nucleotide_alphabet, do_lower_case=True).generate(word_length_vocab)
        with open(vocab_file, 'w') as fout:
            dna_vocab.save(fout)
    tokenizer = DNABertTokenizer(vocab_file, word_length=word_length_vocab, stride=1, do_lower_case=True)

    #
    # Construct the data collator with random masking probability
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=masking_probability)

if __name__ == '__main__':
    main('./data')