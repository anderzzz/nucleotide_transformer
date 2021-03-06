'''Bla bla

'''
from Bio.SeqRecord import SeqRecord
from torch.utils.data import DataLoader

from biosequences.utils import Phrasifier
from biosequences.io import NucleotideSequenceDataset, NucleotideSequenceProcessor

TEST_DATA = './tests_data/'
SEQ_ON_RECORD = 'AGGACGAACGCTGGCGGCGTGCTTAACACATGCAAGTCGAACGAGAGGACATGAAAAGCTTGCTTTTTATGAAATCTAGTGGCAAACGGGTGAGTAAC' + \
                'ACGTAAACAACCTGCCTTCAAGATGGGGACAACAGACGGAAACGACTGCTAATACCGAATACGATCCGAAAGTCGCATGACATTCGGATGAAAGGGTG' + \
                'GCCTATCGAAGAAGCTATCGCTTGAAGAGGGGTTTGCGTCCGATTAGGTAGTTGGTGAGGTAACGGCCCACCAAGCCGACGATCGGTAGCCGGTCTGA' + \
                'GAGGATGAACGGCCACACTGGAACTGAGACACGGTCCAGACTCCTACGGGAGGCAGCAGTGGGGAATCTTCCGCAATGGACGAAAGTCTGACGGAGCA' + \
                'ACGCCGCGTGAGTGAAGACGGCCTTCGGGTTGTAAAGCTCTGTGATTCGGGACGAAAGGCCATATGTGAATAATATATGGAAATGACGGTACCGAAAA' + \
                'AGCAAGCCACGGCTAACTACGTGCCAGCAGCCGCGGTAATACGTAGGTGGCAAGCGTTGTCCGGAATTATTGGGCGTAAAGCGCGCGCAGGCGGTCTC' + \
                'TTAAGTCCATCTTAGAAGTGCGGGGCTTAACCCCGTGAGGGGATGGAAACTGGGAGACTGGAGTATCGGAGAGGAAAGTGGAATTCCTAGTGTAGCGG' + \
                'TGAAATGCGTAGATATTAGGAAGAACACCGGTGGCGAAGGCGACTTTCTGGACGAAAACTGACGCTGAGGCGCGAAAGCGTGGGGAGCAAACAGGATT' + \
                'AGATACCCTGGTAGTCCACGCCGTAAACGATGGATACTAGGTGTAGGAGGTATCGACCCCTTCTGTGCCGGAGTTAACGCAATAAGTATCCCGCCTGG' + \
                'GAAGTACGATCGCAAGATTAAAACTCAAAGGAATTGACGGGGGCCCGCACAAGCGGTGGAGTATGTGGTTTAATTCGACGCAACGCGAAGAACCTTAC' + \
                'CAAGTCTTGACATTGATCGCCATTCCAAGAGATTGGAAGTTCTCCTTCGGGAGACGAGAAAACAGGTGGTGCACGGCTGTCGTCAGCTCGTGTCGTGA' + \
                'GATGTTGGGTTAAGTCCCGCAACGAGCGCAACCCCTATCTTTTGTTGCCAGCACGTAGAGGTGGGAACTCAGAAGAGACCGCCGCAGACAATGCGGAG' + \
                'GAAGGTGGGGATGACGTCAAGTCATCATGCCCCCTATGACCTGGGCTACACACGTGCTACAATGGACGGTACAACGAGAAGCGACCCTGTGAAGGCAA' + \
                'GCGGATCTCTGAAAGCCGTTCTCAGTTCGGATTGCAGGCTGCAACTCGCCTGCATGAAGCTGGAATCGCTAGTAATCGCAAATCAGCACGTTGCGGTG' + \
                'AATACGTTCCCGGGCCTTGTACACACCGCCCGTCACACCA'
SEQ_ON_RECORD_PHRASE_LEN = 4 * (len(SEQ_ON_RECORD) - 2) - 1


def test_simple_read():
    data = NucleotideSequenceDataset(source_files=['{}/jx871316.gb'.format(TEST_DATA)],
                                     source_file_format='genbank')
    assert len(data) == 1

    dloader = DataLoader(data)
    for dd in dloader:
        assert dd['seq'][0] == SEQ_ON_RECORD

def test_phrase_read():
    phraser = Phrasifier(stride=1, word_length=3)
    data = NucleotideSequenceDataset(source_files=['{}/jx871316.gb'.format(TEST_DATA)],
                                     source_file_format='genbank', phrasifier=phraser)
    assert len(data) == 1

    dloader = DataLoader(data)
    for dd in dloader:
        assert len(dd['seq'][0]) == SEQ_ON_RECORD_PHRASE_LEN
        assert dd['seq'][0][-3:] == SEQ_ON_RECORD[-3:]
        assert dd['seq'][0][:3] == SEQ_ON_RECORD[:3]

def test_convert_2json():
    phraser = Phrasifier(stride=1, word_length=3)
    converter = NucleotideSequenceProcessor(source_directory=TEST_DATA,
                                            source_file_format='genbank',
                                            source_directory_file_pattern='*.gb')
    converter.save_as_json(save_dir=TEST_DATA, seq_transformer=phraser)

def test_convert_2json2():
    phraser = Phrasifier(stride=1, word_length=3, do_upper_case=True)
    p = NucleotideSequenceProcessor(source_directory=TEST_DATA,
                                    source_directory_file_pattern='test*.csv',
                                    source_file_format='csv')
    p.save_as_json(save_dir=TEST_DATA, save_prefix='dummy', seq_transformer=phraser)

if __name__ == '__main__':
    test_simple_read()
    test_phrase_read()
    test_convert_2json()
    test_convert_2json2()