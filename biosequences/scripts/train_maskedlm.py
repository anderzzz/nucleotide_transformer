'''Bla bla

'''
from pathlib import Path

from biosequences.io import NucleotideSequenceProcessor
from biosequences.utils import NucleotideVocabCreator, dna_nucleotide_alphabet, Phrasifier
from biosequences.datacollators import DataCollatorDNAWithMasking

from transformers import BertForMaskedLM, BertConfig
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
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

def fit_bert_maskedlm(folder_seq_raw=None, seq_raw_format='csv', seq_raw_file_pattern='*.csv', upper_lower='upper',
                      folder_seq_sentence=None, seq_sentence_prefix='',
                      vocab_file='vocab.txt', create_vocab=True, word_length_vocab=3, stride=1,
                      chunk_size=1000,
                      masking_probability=0.15,
                      bert_config_kwargs={},
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

    if folder_training_output is None:
        folder_training_output_ = folder_seq_sentence
    else:
        folder_training_output_ = folder_training_output

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
    # Tokenize the dataset and make compatible with Huggingface data loaders
    json_files = Path(folder_seq_sentence).glob('{}*.json'.format(seq_sentence_prefix))
    json_files = ['{}'.format(x.resolve()) for x in json_files]
    seq_dataset = load_dataset('json',
                               data_files=json_files)
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
    config = BertConfig(**bert_config_kwargs)
    model = BertForMaskedLM(config=config)

    #
    # Set up trainer
    training_args = TrainingArguments(
        output_dir=folder_training_output_,
        overwrite_output_dir=False,
        num_train_epochs=1,
        per_gpu_train_batch_size=64,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lm_dataset['train']
    )

    #
    # Now train.
    trainer.train()
