'''Bla bla

'''
from pathlib import Path

from biosequences.io import NucleotideSequenceProcessor
from biosequences.tokenizers import DNABertTokenizer
from biosequences.utils import NucleotideVocabCreator, dna_nucleotide_alphabet, Phrasifier
from biosequences.datacollators import DataCollatorDNAWithMasking

from transformers import BertForMaskedLM, BertConfig
from transformers import BertTokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

def sequence_grouper(seqs, chunk_size):
    concat_seq = {k : sum(seqs[k], []) for k in seqs.keys()}
    total_length = len(concat_seq[list(seqs.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    result = {
        k : [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concat_seq.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result

def main(gb_data_folder=None, out_data_folder='/data_out',
         vocab_file='vocab.txt', create_vocab=True, word_length_vocab=3, stride=1,
         chunk_size=1000,
         masking_probability=0.15):

    #
    # Construct Dataset
    phrasifier = Phrasifier(stride=stride, word_length=word_length_vocab)
    if gb_data_folder is None:
        raise ValueError('The folder with Genbank data files must be provided in `gb_data_folder`')
    dataprocessor = NucleotideSequenceProcessor(gb_data_folder,
                                                phrasifier=phrasifier,
                                                attributes_extract=('seq',))
    dataprocessor.convert_to_('json', out_dir=out_data_folder)

    #
    # Construct the tokenizer
    if create_vocab:
        dna_vocab = NucleotideVocabCreator(dna_nucleotide_alphabet, do_lower_case=True).generate(word_length_vocab)
        with open('{}/{}'.format(out_data_folder, vocab_file), 'w') as fout:
            dna_vocab.save(fout)
    tokenizer = BertTokenizer(vocab_file='{}/{}'.format(out_data_folder, vocab_file))

    #
    # Tokenize the dataset and make compatible with Huggingface data loaders
    json_files = Path(out_data_folder).glob('*.json')
    json_files = ['{}'.format(x.resolve()) for x in json_files]
    seq_dataset = load_dataset('json',
                               data_files=json_files)
    tokenized_dataset = seq_dataset.map(
        lambda x: tokenizer(x['seq']), batched=True, remove_columns=['seq']
    )
    lm_dataset = tokenized_dataset.map(
        sequence_grouper,
        batched=True,
        fn_kwargs={'chunk_size' : chunk_size}
    )
    print (lm_dataset)

    #
    # Construct the data collator with random masking probability
#    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=masking_probability)
    # The Data Collator should return dictionary that can be put into model as kwargs
    data_collator = DataCollatorDNAWithMasking(tokenizer=tokenizer,
                                               mlm_probability=masking_probability,
                                               word_length=word_length_vocab)
    print (data_collator)
    ##
    ## HERE!!!
    ## WORRY: IF FIRST TOKEN MASKED IN A GROUP CANNOT EXPAND TO NEXT GROUP. PREVENT ALL INTERGROUP MASKS??
#    xx = data_collator([lm_dataset['train'][k] for k in range(4)], return_tensors='pt')
    xx = data_collator([lm_dataset['train'][1]], return_tensors='pt')
    print (xx)
    print (xx['input_ids'])

    config = BertConfig()
    model = BertForMaskedLM(config=config)

#    model(**xx)

    training_args = TrainingArguments(
        output_dir=out_data_folder,
        overwrite_output_dir=False,
        num_train_epochs=1,
        per_gpu_train_batch_size=64,
        save_steps=10000,
        save_total_limit=2,
        prediction_loss_only=True
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=lm_dataset['train']
    )

    trainer.train()


if __name__ == '__main__':
    main(gb_data_folder='/Users/andersohrn/PycharmProjects/nucleotide_transformer/biosequences/scripts/data',
         out_data_folder='/Users/andersohrn/PycharmProjects/nucleotide_transformer/biosequences/scripts/data_out')