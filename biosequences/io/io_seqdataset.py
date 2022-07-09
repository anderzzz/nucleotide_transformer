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

class NucleotideSequenceProcessorIndexException(Exception):
    pass

def _read_seq_file_(fp, format='genbank'):
    for record in SeqIO.parse(fp, format):
        yield record

def _read_seq_csv_(fp, index_col=None, key='sequence_chunk', description=''):
    df = pd.read_csv(fp, index_col=index_col)
    for index, value in df.iterrows():
        yield SeqRecord(Seq(value[key]), id=str(index), name=str(fp), description=description)

def _seqrecord2dict_(record):
    return {
        'seq' : record.seq.__str__(),
        'id' : record.id.__str__(),
        'name' : record.name.__str__(),
        'description' : record.description.__str__()
    }

class NucleotideSequenceProcessor(object):
    '''Bla bla

    '''
    def __init__(self,
                 source_directory=None,
                 source_directory_file_pattern='*',
                 source_files=None,
                 source_file_format='csv',
                ):

        if not source_directory is None:
            p = Path(source_directory)
            if not p.is_dir():
                raise ValueError('File folder {} not found'.format(source_directory))
            self.file_paths = list(p.glob(source_directory_file_pattern))

            if len(self.file_paths) == 0:
                raise ValueError('Zero file paths found in directory {} and query {}'.format(source_directory, source_directory_file_pattern))

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

    def process_file_(self, k_file):
        '''Bla bla

        '''
        try:
            fp = self.file_paths[k_file]
        except IndexError:
            raise NucleotideSequenceProcessorIndexException(
                'File of index {} exceed number of files {}'.format(k_file, len(self.file_paths))
            )

        return list(self.source_reader(fp, **self.source_reader_kwargs))

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
            record_data = _seqrecord2dict_(record)
            record_data['seq'] = _transform(record_data['seq'], **seq_transformer_kwargs)
            with open('{}/{}{}.json'.format(save_dir, save_prefix, k_record), 'w') as f_json:
                json.dump(record_data, indent=4, fp=f_json)


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
    def __init__(self,
                 source_directory=None,
                 source_directory_file_pattern='*',
                 source_files=None,
                 source_file_format='csv',
                 phrasifier=None,
                 phrasifier_kwargs={}
                 ):
        super(NucleotideSequenceDataset, self).__init__()

        self.processor = NucleotideSequenceProcessor(source_directory=source_directory,
                                                     source_directory_file_pattern=source_directory_file_pattern,
                                                     source_files=source_files,
                                                     source_file_format=source_file_format)

        #
        # Map each sequence ID to an integer index. This requires reading all files because some
        # file formats, like Genbank, can have a variable number of sequences per file
        self.seq_mapper = {}
        index = 0
        fp_counter = 0
        while True:
            try:
                data = self.processor.process_file_(fp_counter)
            except NucleotideSequenceProcessorIndexException:
                break

            for ind in range(len(data)):
                self.seq_mapper[index + ind] = (fp_counter, ind)

            index += len(data)
            fp_counter += 1

        self.n_file_paths = index

        self.transform_seq_kwargs = phrasifier_kwargs
        if phrasifier is None:
            self.transform_seq = lambda x : x
        elif callable(phrasifier):
            self.transform_seq = phrasifier
        else:
            raise ValueError('Sequence phrasifier must be callable')


    def __len__(self):
        return self.n_file_paths

    def __getitem__(self, item):
        fp_counter, record_id = self.seq_mapper[item]
        record = self.processor.process_file_(fp_counter)[record_id]
        record = _seqrecord2dict_(record)
        record['seq'] = self.transform_seq(record['seq'], **self.transform_seq_kwargs)

        return record
