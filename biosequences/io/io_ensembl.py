'''Bla bla

'''
import requests
from pathlib import Path
from collections import OrderedDict

from dataclasses import dataclass
from typing import Dict

ENSEMBLE_SERVER = "https://rest.ensembl.org"

@dataclass
class EnsembleAPI:
    server : str
    endpoint : str
    query : Dict

ensemble_sequence_id_ret_json = EnsembleAPI(ENSEMBLE_SERVER,
                                            "sequence/id/",
                                            {'content-type' : 'application/json'})
ensemble_sequence_lookup_ret_json = EnsembleAPI(ENSEMBLE_SERVER,
                                            "lookup/id/",
                                            {'expand' : '1', 'content-type' : 'application/json'})

def _one_slash(part):
    return '{}/'.format(part) if not part.endswith('/') else '{}'.format(part)

class _Requestor(object):
    '''Bla bla

    '''
    def __init__(self, ensembl_api):
        self.ensembl_api = ensembl_api
        if len(self.ensembl_api.query) == 0:
            self.add_query = False
        else:
            self.add_query = True

    def _make_base(self):
        return _one_slash(self.ensembl_api.server) + _one_slash(self.ensembl_api.endpoint)

    def generate_url_(self, items=[]):
        url = self._make_base()
        for item in items:
            url_full = '{}{}'.format(url, item)
            if self.add_query:
                url_full += '?'
                for query_key, query_val in self.ensembl_api.query.items():
                    url_full += '{}={};'.format(query_key, query_val)
                url_full = url_full[:-1]

            yield url_full

    def retrieve(self, items):
        for url in self.generate_url_(items):
            response = requests.get(url)
            yield response

class SequenceChunkMaker(object):
    '''Bla bla

    '''
    def __init__(self, save_dir='.',
                 sequence_file_name='sequences.csv',
                 exon_intron_file_name='exon_intron_labels.csv',
                 metadata_file_name='metadata.csv',
                 transcript_only=True,
                 yield_data=False
                 ):

        self.yield_data = yield_data
        self.save_dir = save_dir
        self.metadata_file_name = metadata_file_name

        self.sequence_file_name = sequence_file_name
        self.exon_intron_file_name = exon_intron_file_name
        if not self.yield_data:
            metadata_file = Path('{}/{}'.format(self.save_dir, self.metadata_file_name))
            metadata_file.unlink()
            with open('{}/{}'.format(save_dir, sequence_file_name), 'w') as fout:
                print('id,nuc_sequence', file=fout)
            with open('{}/{}'.format(save_dir, exon_intron_file_name), 'w') as fout:
                print('id,label_sequence', file=fout)

        self.transcript_only = transcript_only

        self.seq_requestor = _Requestor(ensembl_api=ensemble_sequence_id_ret_json)
        self.seq_lookupor = _Requestor(ensembl_api=ensemble_sequence_lookup_ret_json)

    def __call__(self, seq_ids):
        for seq_response, lookup_response in zip(self.seq_requestor.retrieve(seq_ids),
                                                 self.seq_lookupor.retrieve(seq_ids)):

            if seq_response.status_code != 200:
                continue

            seq_data = seq_response.json()
            lookup_data = lookup_response.json()

            if self.transcript_only and not lookup_data['object_type'] == 'Transcript':
                raise RuntimeError('Encountered ID {}, which is not a transcript object'.format(seq_data['id']))

            full_sequence = seq_data['seq']
            label_sequence = ['intron'] * len(full_sequence)
            end_sequence = lookup_data['end']

            n_exons = 0
            for exon in lookup_data['Exon']:
                start_renorm = end_sequence - exon['end']
                end_renorm = end_sequence - exon['start'] + 1
                label_sequence[start_renorm : end_renorm] = [exon['id']] * (end_renorm - start_renorm)
                n_exons += 1

            metadata = OrderedDict([('sequence_id', seq_data['id']),
                                    ('version', seq_data['version']),
                                    ('species', lookup_data['species']),
                                    ('display_name', lookup_data['display_name']),
                                    ('n_exons', n_exons)])
            if not self.yield_data:
                self.save(metadata=metadata,
                          nuc_sequence=full_sequence,
                          exon_intron=label_sequence)
            else:
                yield metadata, full_sequence, label_sequence

    def save(self, metadata, nuc_sequence, exon_intron):
        '''Bla bla

        '''
        metadata_file = Path('{}/{}'.format(self.save_dir, self.metadata_file_name))
        if metadata_file.is_file():
            line = ','.join([str(x) for x in metadata.values()])
            with open('{}/{}'.format(self.save_dir, self.metadata_file_name), 'a') as fout:
                print('{}'.format(line), file=fout)
        else:
            line_header = ','.join([str(x) for x in metadata.keys()])
            line = ','.join([str(x) for x in metadata.values()])
            with open('{}/{}'.format(self.save_dir, self.metadata_file_name), 'w') as fout:
                print('{}'.format(line_header), file=fout)
                print('{}'.format(line), file=fout)

        with open('{}/{}'.format(self.save_dir, self.sequence_file_name), 'a') as fout:
            print ('{},{}'.format(metadata['sequence_id'], nuc_sequence), file=fout)
        with open('{}/{}'.format(self.save_dir, self.exon_intron_file_name), 'a') as fout:
            print ('{},{}'.format(metadata['sequence_id'], exon_intron), file=fout)
