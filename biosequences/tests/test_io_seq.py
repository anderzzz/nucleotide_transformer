'''Bla bla

'''
from Bio.SeqRecord import SeqRecord
from torch.utils.data import DataLoader

from biosequences import NucleotideSequenceDataset

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


def test_simple_read():
    data = NucleotideSequenceDataset(TEST_DATA)
    assert len(data) == 1

    dloader = DataLoader(data)
    for dd in dloader:
        assert dd[0][0] == SEQ_ON_RECORD

def test_full_record_read():
    data = NucleotideSequenceDataset(TEST_DATA, output_record_filter_func='full_record')
    assert len(data) == 1

    assert isinstance(data[0], SeqRecord)
    assert data[0].seq.__str__() == SEQ_ON_RECORD

if __name__ == '__main__':
    test_simple_read()
    test_full_record_read()