from train_maskedlm import fit_bert_maskedlm
from eval_maskedlm import eval_bert_maskedlm

eval_bert_maskedlm(
    folder_seq_raw=None,
    folder_seq_sentence='/Users/andersohrn/Development/DNABert/models/lm_model_tmp/data_tmp',
    word_length_vocab=3,
    masking_probability=0.15,
    folder_training_input='/Users/andersohrn/Development/DNABert/models/lm_model/',
    bert_config_kwargs={
        'max_position_embeddings' : 1024
    }
)

quit()

fit_bert_maskedlm(
#    folder_seq_raw='/Users/andersohrn/Development/DNABert/data/lm_data/',
    seq_raw_format='csv',
    seq_raw_file_pattern='grch38_p14_chunks.csv',
    upper_lower='upper',
    folder_seq_sentence='/Users/andersohrn/Development/DNABert/models/lm_model_tmp/data_tmp',
    word_length_vocab=3,
    chunk_size=1000,
    shuffle=True,
    split_ratio_test=0.90, split_ratio_validate=0.020,
    masking_probability=0.15,
    folder_training_input='/Users/andersohrn/Development/DNABert/models/lm_model/',
    folder_training_output='/Users/andersohrn/Development/DNABert/models/lm_model_tmp/',
    bert_config_kwargs={
        'max_position_embeddings' : 1024
    },
    lr=0.01,
    weight_decay=0.1,
    n_warmup_steps=100,
    training_kwargs={
        'fp16' : False,
        'per_device_train_batch_size' : 2,
        'gradient_accumulation_steps' : 8,
        'num_train_epochs' : 5,
        'save_steps' : 100,
        'evaluation_strategy' : 'epoch'
    }
)