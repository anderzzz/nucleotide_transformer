'''Bla bla

'''
import requests

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
    def __init__(self, save_dir=None):
        self.save_dir = save_dir

        self.seq_requestor = _Requestor(ensembl_api=ensemble_sequence_id_ret_json)
        self.seq_lookupor = _Requestor(ensembl_api=ensemble_sequence_lookup_ret_json)

    def __call__(self, seq_ids):
        for seq_response, lookup_response in zip(self.seq_requestor.retrieve(seq_ids),
                                                 self.seq_lookupor.retrieve(seq_ids)):

            if seq_response.status_code != 200:
                continue

            seq_data = seq_response.json()
            lookup_data = lookup_response.json()

            full_sequence = seq_data['seq']
            label_sequence = ['intron'] * len(full_sequence)
            start_sequence = lookup_data['start']
            end_sequence = lookup_data['end']
            species = lookup_data['species']

            n_exons = 0
            for exon in lookup_data['Exon']:
                start_renorm = end_sequence - exon['end']
                end_renorm = end_sequence - exon['start'] + 1
                label_sequence[start_renorm : end_renorm] = [exon['id']] * (end_renorm - start_renorm)
                n_exons += 1

            self.save(metadata={'sequence_id': seq_data['id'],
                                'version' : seq_data['version'],
                                'species' : lookup_data['species'],
                                'display_name' : lookup_data['display_name'],
                                'n_exons' : n_exons},
                      nuc_sequence=full_sequence,
                      exon_intron=label_sequence)

    def save(self, metadata, nuc_sequence, exon_intron):
        pass

aa = SequenceChunkMaker()
aa(['ENSOCUT00000001061', 'ENSOCUT00000001061XXXX#'])

pp = _Requestor(ensemble_sequence_id_ret_json)
rr = _Requestor(ensemble_sequence_lookup_ret_json)
list(pp.retrieve(['ENSOCUT00000001061XXXX#']))
list(rr.retrieve(['ENSOCUT00000001061']))
