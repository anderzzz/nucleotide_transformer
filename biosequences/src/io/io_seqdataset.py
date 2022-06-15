'''Bla bla

'''
from pathlib import Path
from torch.utils.data import Dataset
from Bio import SeqIO

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
                 output_record_filter_func='seq_with_id'):
        super(NucleotideSequenceDataset, self).__init__()

        p = Path(source_directory)
        if not p.is_dir():
            raise ValueError('File folder {} not found'.format(source_directory))

        self.file_paths = list(p.glob(input_file_pattern))
        self.input_seqio_format = input_seqio_format

        if output_record_filter_func == 'seq_with_id':
            self.filter_record_ = self._filter_record_seq_with_id
        elif output_record_filter_func == 'full_record':
            self.filter_record_ = lambda x: x
        elif callable(output_record_filter_func):
            self.filter_record_ = output_record_filter_func
        else:
            raise ValueError('Unknown value for record filter parameter: {}'.format(output_record_filter_func))

        #
        # Map each sequence ID to an integer index. This requires reading all files because some
        # file formats, like Genbank, can have a variable number of sequences per file
        self.seq_mapper = {}
        index = 0
        for file_path in self.file_paths:
            records = self._read_file_(file_path)
            for record in records:
                self.seq_mapper[index] = (file_path, record.id)
                index += 1

        self.n_file_paths = index

    def __len__(self):
        return self.n_file_paths

    def __getitem__(self, item):
        fp, record_id = self.seq_mapper[item]
        records = self._read_file_(fp, format_name=self.input_seqio_format)

        # The SeqRecord index in case of multiple sequences per file is 1-based
        counter = int(record_id.split('.')[-1])
        record = records[counter - 1]

        return self.filter_record_(record)

    def _read_file_(self, fp, format_name='genbank'):
        return [record for record in SeqIO.parse(fp, format_name)]

    def _filter_record_seq_with_id(self, record):
        return record.seq.__str__(), record.id, record.name