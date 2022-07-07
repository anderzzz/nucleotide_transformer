from biosequences.io.io_seqdataset import (
    NucleotideSequenceDataset,
    NucleotideSequenceProcessor
)

from biosequences.io.io_ensembl import (
    SequenceChunkMaker
)

from biosequences.utils import _Factory
class IOFactory(_Factory):
    '''Bla bla

    '''
    def __init__(self):
        super(IOFactory, self).__init__()

factory = IOFactory()
factory.register_builder('genbank')