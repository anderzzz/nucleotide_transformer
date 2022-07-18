'''Bla bla

'''
from pathlib import Path
import random
import torch

from biosequences.io import NucleotideSequenceProcessor
from biosequences.utils import NucleotideVocabCreator, dna_nucleotide_alphabet, Phrasifier, factory_lr_schedules
from biosequences.datacollators import DataCollatorDNAWithMasking

from transformers import BertForMaskedLM, BertConfig
from transformers import BertTokenizer
from datasets import load_dataset

def _sequence_grouper(seqs, chunk_size):
    concat_seq = {k : sum(seqs[k], []) for k in seqs.keys()}
    total_length = len(concat_seq[list(seqs.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    result = {
        k : [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concat_seq.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result

def eval_bert_maskedlm(folder_seq_raw=None, seq_raw_format='csv', seq_raw_file_pattern='*.csv', upper_lower='upper',
                      folder_seq_sentence=None, seq_sentence_prefix='',
                      vocab_file='vocab.txt', create_vocab=True, word_length_vocab=3, stride=1,
                      masking_probability=0.15,
                      bert_config_kwargs={},
                      folder_training_input=None,
                      folder_training_output=None):

    #
    # Some argument sanity checks
    if folder_seq_sentence is None:
        raise ValueError('The folder for sequence sentence files required')

    if upper_lower == 'upper':
        do_upper_case = True
        do_lower_case = False
    elif upper_lower == 'lower':
        do_upper_case = False
        do_lower_case = True
    else:
        raise ValueError('The vocabulary is either all upper or all lower, but `upper_lower` of invalid value: {}'.format(upper_lower))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #
    # Construct sequence sentences from sequence data if such folder is given
    if not folder_seq_raw is None:
        dataprocessor = NucleotideSequenceProcessor(source_directory=folder_seq_raw,
                                                    source_file_format=seq_raw_format,
                                                    source_directory_file_pattern=seq_raw_file_pattern)
        phrasifier = Phrasifier(stride=stride,
                                word_length=word_length_vocab,
                                do_upper_case=do_upper_case,
                                do_lower_case=do_lower_case)
        dataprocessor.save_as_json(save_dir=folder_seq_sentence,
                                   save_prefix=seq_sentence_prefix,
                                   seq_transformer=phrasifier)

    #
    # Construct the tokenizer
    if create_vocab:
        dna_vocab = NucleotideVocabCreator(alphabet=dna_nucleotide_alphabet,
                                           do_lower_case=do_lower_case,
                                           do_upper_case=do_upper_case).generate(word_length_vocab)
        with open('{}/{}'.format(folder_seq_sentence, vocab_file), 'w') as fout:
            dna_vocab.save(fout)
    tokenizer = BertTokenizer(vocab_file='{}/{}'.format(folder_seq_sentence, vocab_file), do_lower_case=do_lower_case)

    #
    # Load dataset
    json_files = Path(folder_seq_sentence).glob('{}*.json'.format(seq_sentence_prefix))
    json_files = ['{}'.format(x.resolve()) for x in json_files]
    json_files = json_files[:2]
    seq_dataset = load_dataset('json',
                               data_files=json_files)

    #
    # Tokenize the data and make compatible with huggingsface collator
    tokenized_dataset = seq_dataset.map(
        lambda x: tokenizer(x['seq']), batched=True, remove_columns=['seq', 'id', 'name', 'description']
    )
#    lm_dataset = tokenized_dataset.map(
#        batched=True,
#    )
#    print (lm_dataset)

    #
    # Construct the data collator with random masking probability. Note that the standard Huggingsface method
    # of token masking does not work for DNA tokens of word length greater than one. Hence the custom collator
    data_collator = DataCollatorDNAWithMasking(tokenizer=tokenizer,
                                               mlm_probability=masking_probability,
                                               word_length=word_length_vocab)

    #
    # Configure the model for masked language modelling. If no model input directory is given,
    # then configure from scratch. Else, load model, its configuration and parameters, from input folder data.
    # Note that the loading of input assumes certain file naming, which standard training ensures.
    if folder_training_input is None:
        config = BertConfig(vocab_size=tokenizer.vocab_size,
                            **bert_config_kwargs)
        model = BertForMaskedLM(config=config)
    else:
        model = BertForMaskedLM.from_pretrained(folder_training_input)
    model = model.to(device)

    #
    # Evaluate on some input and show top five
    print (tokenized_dataset['train'])
    inp_data_masked = data_collator([tokenized_dataset['train'][0]])
    out = model(**inp_data_masked)
    print (out)
    print ('jaaa')
