'''Bla bla

'''
from pandas import read_csv

from biosequences.io import SequenceChunkMaker

def _parse_g3po(fin):
    df = read_csv(filepath_or_buffer=fin, sep=';')
    df = df.loc[df['Quality'] == 'Confirmed']  # High quality selection
    df = df.loc[df['Transcrit_ID'].str.slice(0, 2) == 'EN']  # Only those that are in Ensembl database
    return df['Transcrit_ID'].to_list()

def make_exonintron_db(
        save_dir='.',
        transcript_ids=None,
        transcript_ids_file=None,
        transcript_ids_file_parser=None,
        transcript_ids_file_parser_kwargs={},
        compact_label=False
    ):

    sequence_chunk_maker = SequenceChunkMaker(save_dir=save_dir, yield_data=False, compact_label=compact_label)
    if transcript_ids is None:
        if transcript_ids_file is None:
            raise ValueError('One of `transcript_ids` and `transcript_ids_file` has to be other than `None`')

        with open(transcript_ids_file) as fin:
            transcript_ids = transcript_ids_file_parser(fin, **transcript_ids_file_parser_kwargs)

    print ('Total transcript IDs: {}'.format(len(transcript_ids)))
    sequence_chunk_maker(transcript_ids)


if __name__ == '__main__':
    make_exonintron_db(save_dir='.',
                       transcript_ids_file='/Users/andersohrn/PycharmProjects/nucleotide_transformer/data/G3PO+.csv',
                       transcript_ids_file_parser=_parse_g3po,
                       transcript_ids_file_parser_kwargs={},
                       compact_label=False
                       )