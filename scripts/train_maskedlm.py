'''Bla bla

'''
from pathlib import Path
import random
import numpy as np

from biosequences.io import NucleotideSequenceProcessor
from biosequences.utils import NucleotideVocabCreator, dna_nucleotide_alphabet, Phrasifier
from biosequences.datacollators import DataCollatorDNAWithMasking

from transformers import BertForMaskedLM, BertConfig
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_metric

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

def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    matches = [x == y for x, y in zip(predictions.flatten(), labels.flatten()) if y > -99]
    return {'match_percentage' : 100.0 * matches.count(True) / len(matches)}

def fit_bert_maskedlm(folder_seq_raw=None, seq_raw_format='csv', seq_raw_file_pattern='*.csv', upper_lower='upper',
                      folder_seq_sentence=None, seq_sentence_prefix='',
                      split_ratio_test=0.05, split_ratio_validate=0.05, shuffle=True, seed=42,
                      vocab_file='vocab.txt', create_vocab=True, word_length_vocab=3, stride=1,
                      chunk_size=1000,
                      masking_probability=0.15,
                      bert_config_kwargs={},
                      folder_training_output=None,
                      training_kwargs={}):

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

    if folder_training_output is None:
        folder_training_output_ = folder_seq_sentence
    else:
        folder_training_output_ = folder_training_output

    random.seed(seed)

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
    # Create dataset appropriately split from the sequence sentences stored in the json files
    json_files = Path(folder_seq_sentence).glob('{}*.json'.format(seq_sentence_prefix))
    json_files = ['{}'.format(x.resolve()) for x in json_files]
    len_train = round(len(json_files) * (1.0 - split_ratio_test - split_ratio_validate))
    if len_train <= 0:
        raise ValueError('Split ratios for test and validate exceed 1.0, leaving nothing for training')
    len_test = round(len(json_files) * split_ratio_test)
    pp = list(range(len(json_files)))
    if shuffle:
        random.shuffle(pp)
    json_files_split = {'train' : [json_files[k] for k in pp[:len_train]]}
    if len_test > 0:
        json_files_split['test'] = [json_files[k] for k in pp[len_train:len_train + len_test]]
    if len(json_files) - len_train - len_test > 0:
        json_files_split['validate'] = [json_files[k] for k in pp[len_train + len_test:]]
    json_files = json_files_split
    seq_dataset = load_dataset('json',
                               data_files=json_files)

    #
    # Tokenize the data and make compatible with huggingsface collator
    tokenized_dataset = seq_dataset.map(
        lambda x: tokenizer(x['seq']), batched=True, remove_columns=['seq', 'id', 'name', 'description']
    )
    lm_dataset = tokenized_dataset.map(
        _sequence_grouper,
        batched=True,
        fn_kwargs={'chunk_size' : chunk_size}
    )

    #
    # Construct the data collator with random masking probability. Note that the standard Huggingsface method
    # of token masking does not work for DNA tokens of word length greater than one. Hence the custom collator
    data_collator = DataCollatorDNAWithMasking(tokenizer=tokenizer,
                                               mlm_probability=masking_probability,
                                               word_length=word_length_vocab)

    #
    # Configure the model for masked language modelling. This is appropriate for training the Bert encoder
    config = BertConfig(vocab_size=tokenizer.vocab_size,
                        **bert_config_kwargs)
    model = BertForMaskedLM(config=config)

    #
    # Set up trainer
    training_args = TrainingArguments(
        output_dir=folder_training_output_,
        **training_kwargs
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lm_dataset['train'],
        eval_dataset=lm_dataset['test'],
        compute_metrics=_compute_metrics
    )

    #
    # Now train.
    trainer.train()
    trainer.save_model(output_dir=folder_training_output_)
    print (trainer)

    print ('hellow')
