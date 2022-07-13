from train_maskedlm import fit_bert_maskedlm

fit_bert_maskedlm(
#    folder_seq_raw='/Users/andersohrn/Development/DNABert/data/lm_data/',
    seq_raw_format='csv',
    seq_raw_file_pattern='grch38_p14_chunks.csv',
    upper_lower='upper',
    folder_seq_sentence='/Users/andersohrn/Development/DNABert/models/lm_model_tmp/data_tmp',
    word_length_vocab=3,
    chunk_size=1024,
    split_ratio_test=0.95, split_ratio_validate=0.005,
    masking_probability=0.15,
    folder_training_input='/Users/andersohrn/Development/DNABert/models/lm_model_tmp/',
    folder_training_output='/Users/andersohrn/Development/DNABert/models/lm_model_tmp/',
    bert_config_kwargs={
        'max_position_embeddings' : 1024
    },
    training_kwargs={
        'per_device_train_batch_size' : 2,
        'num_train_epochs' : 2,
        'save_steps' : 30,
        'evaluation_strategy' : 'epoch'
    }
)