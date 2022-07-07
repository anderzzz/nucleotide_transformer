'''Bla bla

'''
from pathlib import Path
import json
from collections.abc import Iterable
import pandas as pd

from torch.utils.data import Dataset
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

def _read_seq_file_(fp, format_name='genbank'):
    for record in SeqIO.parse(fp, format_name):
        yield record

def _read_seq_csv_(fp, index_col=None, key='sequence_chunk', description=''):
    df = pd.read_csv(fp, index_col=index_col)
    for index, value in df.iterrows():
        yield SeqRecord(Seq(value[key]), id=str(index), name=str(fp), description=description)

class NucleotideSequenceProcessor(object):
    '''Bla bla

    '''
    def __init__(self,
                 source_directory=None,
                 source_directory_file_pattern='*',
                 source_files=None,
                 source_file_format='csv'
                ):

        if not source_directory is None:
            p = Path(source_directory)
            if not p.is_dir():
                raise ValueError('File folder {} not found'.format(source_directory))
            self.file_paths = list(p.glob(source_directory_file_pattern))

        elif not source_files is None:
            if isinstance(source_files, Iterable):
                self.file_paths = source_files
            else:
                raise ValueError('The `source_files` is not an iterable, rather: {}'.format(type(source_files)))

        else:
            raise ValueError('Failed to specify raw data source files')

        if source_file_format == 'csv':
            self.source_reader = _read_seq_csv_
            self.source_reader_kwargs = {'index_col' : 0}
        elif source_file_format in ['genbank', 'fasta']:
            self.source_reader = _read_seq_file_
            self.source_reader_kwargs = {'format' : source_file_format}
        else:
            raise ValueError('Unknown source file format: {}'.format(source_file_format))

    def __iter__(self):
        '''Bla bla

        '''
        for fp in self.file_paths:
            for data_package in self.source_reader(fp, **self.source_reader_kwargs):
                yield data_package

    def save_as_json(self, save_dir=None, save_prefix='', seq_transformer=None, seq_transformer_kwargs={}):
        '''Bla bla

        '''
        if seq_transformer is None:
            _transform = lambda x : x
        elif callable(seq_transformer):
            _transform = seq_transformer
        else:
            raise ValueError('Sequence transformer must be callable')

        for k_record, record in enumerate(self):
            seq_out = _transform(record.seq.__str__(), **seq_transformer_kwargs)
            metadata_id = record.id
            metadata_name = record.name
            metadata_description = record.description
            with open('{}/{}{}.json'.format(save_dir, save_prefix, k_record), 'w') as f_json:
                json.dump({'name': metadata_name,
                           'id': metadata_id,
                           'description': metadata_description,
                           'sequence_data': seq_out},
                          indent=4, fp=f_json)

from biosequences.utils import Phrasifier
p = NucleotideSequenceProcessor(source_directory='/Users/andersohrn/PycharmProjects/nucleotide_transformer',
                                source_directory_file_pattern='test*.csv',
                                source_file_format='csv')
pp = Phrasifier(1,3,True)
p.save_as_json('.',save_prefix='dummy',seq_transformer=pp)

class NucleotideSequenceDataset(Dataset):
    '''Dataset of nucleotide sequences from collection of data in some standard format

    Args:
        source_directory (str): Path to folder from which to retrieve sequence files
        input_file_pattern (str): The glob-style pattern to invoke in the source directory to
            retrieve files containing sequences. Default "*.gb"
        input_seqio_format (str): The format of the sequence files. Same options available as
            for `SeqIO` module of BioPython, see https://biopython.org/wiki/SeqIO. Default "genbank"
        output_record_filter_func (str or callable): How to pre-process the sequence record. If an
            executable its input argument should be an instance of `SeqRecord` of BioPython, and it
            should return the data that should comprise the PyTorch Dataset. Two string options
            available:
            - `seq_with_id` : returns three strings, the sequence, its ID and its name
            - `full_record` : returns the full `SeqRecord` without pre-processing; this option is not
                              compatible with `DataLoader`

    '''
    def __init__(self, source_directory,
                 input_file_pattern='*.gb',
                 input_seqio_format='genbank',
                 attributes_extract=('id', 'name', 'seq'),
                 phrasifier=None
                 ):
        super(NucleotideSequenceDataset, self).__init__()

        self.processor = NucleotideSequenceProcessor(source_directory=source_directory,
                                                     input_file_pattern=input_file_pattern,
                                                     input_seqio_format=input_seqio_format,
                                                     attributes_extract=attributes_extract,
                                                     phrasifier=phrasifier)

        #
        # Map each sequence ID to an integer index. This requires reading all files because some
        # file formats, like Genbank, can have a variable number of sequences per file
        self.seq_mapper = {}
        index = 0
        for fp in self.processor.file_paths:
            records = _read_seq_file_(fp, format_name=input_seqio_format)
            for record in records:
                self.seq_mapper[index] = (fp, record.id)
                index += 1

        self.n_file_paths = index

    def __len__(self):
        return self.n_file_paths

    def __getitem__(self, item):
        fp, record_id = self.seq_mapper[item]
        records = _read_seq_file_(fp, format_name=self.processor.input_seqio_format)

        # The SeqRecord index in case of multiple sequences per file is 1-based
        counter = int(record_id.split('.')[-1])
        record = records[counter - 1]

        return self.processor.extract_(record)
