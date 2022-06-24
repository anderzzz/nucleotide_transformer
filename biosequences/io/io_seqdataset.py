'''Bla bla

'''
from pathlib import Path
import json
from torch.utils.data import Dataset
from Bio import SeqIO

def _read_seq_file_(fp, format_name='genbank'):
    return [record for record in SeqIO.parse(fp, format_name)]

class NucleotideSequenceProcessor(object):
    '''Bla bla

    '''
    def __init__(self,
                 source_directory,
                 input_file_pattern='*.gb',
                 input_seqio_format='genbank',
                 attributes_extract=('id', 'name', 'seq'),
                 phrasifier=None
                 ):

        p = Path(source_directory)
        if not p.is_dir():
            raise ValueError('File folder {} not found'.format(source_directory))

        self.file_paths = list(p.glob(input_file_pattern))
        self.input_seqio_format = input_seqio_format
        self.attributes_extract = attributes_extract
        if phrasifier is None:
            self.phrasify_ = lambda x:x
        elif callable(phrasifier):
            self.phrasify_ = phrasifier
        else:
            raise ValueError('The phrasifier has to be callable or None, not of type {}'.format(type(phrasifier)))

    def convert_to_(self, target_type, **kwargs):
        '''Bla bla

        '''
        if target_type == 'json':
            self._convert_to_json_from_folders_(**kwargs)
        else:
            raise ValueError('Unknown target file type: {}'.format(target_type))

    def _convert_to_json_from_folders_(self, out_dir=None, out_file_suffix='json'):
        '''Bla bla

        '''
        counter = 0
        for fp in self.file_paths:
            records = _read_seq_file_(fp, format_name=self.input_seqio_format)
            for record in records:
                data = self.extract_(record)
                with open('{}/{}.{}'.format(out_dir, counter, out_file_suffix), 'w') as f_json:
                    json.dump(data, indent=4, fp=f_json)

    def extract_(self, record):
        '''Bla bla

        '''
        ret_data = {}
        for attribute in self.attributes_extract:
            val = str(getattr(record, attribute))
            if attribute == 'seq':
                val = self.phrasify_(val)
            ret_data[attribute] = val
        return ret_data

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
