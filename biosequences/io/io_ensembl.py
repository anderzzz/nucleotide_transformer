'''Bla bla

'''
import requests
from pathlib import Path
from collections import OrderedDict
import time

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
ensemble_lookup_metadata_keys = ['display_name', 'db_type', 'version', 'source', 'biotype', 'logic_name', 'species',
                                 'is_canonical']

def _one_slash(part):
    return '{}/'.format(part) if not part.endswith('/') else '{}'.format(part)

class _Requestor(object):
    '''Bla bla

    '''
    def __init__(self, ensembl_api, sleep=0.0):
        self.ensembl_api = ensembl_api
        self.sleep_time = sleep
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
            time.sleep(self.sleep_time)
            print (url)
            response = requests.get(url)
            yield response

class SequenceChunkMaker(object):
    '''Bla bla

    '''
    def __init__(self, save_dir='.',
                 sequence_file_name='sequences_0.csv',
                 exon_intron_file_name='exon_intron_labels_0.csv',
                 metadata_file_name='metadata_0.csv',
                 transcript_only=True,
                 yield_data=False,
                 compact_label=False,
                 error_handler_fn=None,
                 request_sleep=2.0
                 ):

        self.yield_data = yield_data
        self.save_dir = save_dir
        self.metadata_file_name = metadata_file_name
        self.compact_label = compact_label
        if self.compact_label:
            self._label = {'intron' : 'i', 'exon' : 'e'}
        else:
            self._label = {'intron' : 'intron', 'exon' : None}
        if error_handler_fn is None:
            self.error_handler_fn = lambda x, y: None

        self.sequence_file_name = sequence_file_name
        self.exon_intron_file_name = exon_intron_file_name
        if not self.yield_data:
            metadata_file = Path('{}/{}'.format(self.save_dir, self.metadata_file_name))
            metadata_file.unlink(missing_ok=True)
            with open('{}/{}'.format(save_dir, sequence_file_name), 'w') as fout:
                print('id,nuc_sequence', file=fout)
            with open('{}/{}'.format(save_dir, exon_intron_file_name), 'w') as fout:
                print('id,nuc_position,label', file=fout)

        self.transcript_only = transcript_only

        self.seq_requestor = _Requestor(ensembl_api=ensemble_sequence_id_ret_json, sleep=request_sleep)
        self.seq_lookupor = _Requestor(ensembl_api=ensemble_sequence_lookup_ret_json, sleep=request_sleep)

    def __call__(self, seq_ids):
        if self.yield_data:
            return self.process(seq_ids)

        else:
            for metadata, full_sequence, label_sequence in self.process(seq_ids):
                self.save(metadata=metadata,
                          nuc_sequence=full_sequence,
                          exon_intron=label_sequence)
                print ('{}'.format(metadata['sequence_id']))

            return True

    def process(self, seq_ids):
        for seq_response, lookup_response in zip(self.seq_requestor.retrieve(seq_ids),
                                                 self.seq_lookupor.retrieve(seq_ids)):
            if (seq_response.status_code != 200) or (lookup_response.status_code != 200):
                self.error_handler_fn(seq_response, lookup_response)
                continue

            seq_data = seq_response.json()
            lookup_data = lookup_response.json()
            if self.transcript_only and not lookup_data['object_type'] == 'Transcript':
                raise RuntimeError('Encountered ID {}, which is not a transcript object'.format(seq_data['id']))

            full_sequence = seq_data['seq']
            label_sequence = [self._label['intron']] * len(full_sequence)
            end_sequence = lookup_data['end']

            n_exons = 0
            for exon in lookup_data['Exon']:
                start_renorm = end_sequence - exon['end']
                end_renorm = end_sequence - exon['start'] + 1
                if self.compact_label:
                    ie_chunk = [self._label['exon']] * (end_renorm - start_renorm)
                else:
                    ie_chunk = [exon['id']] * (end_renorm - start_renorm)
                label_sequence[start_renorm : end_renorm] = ie_chunk
                n_exons += 1


            metadata = OrderedDict([('sequence_id', seq_data['id']),
                                    ('version_seq_data', seq_data['version']),
                                    ('n_exons', n_exons)])
            for lookup_key in ensemble_lookup_metadata_keys:
                if lookup_key in lookup_data:
                    metadata[lookup_key] = lookup_data[lookup_key]
                else:
                    metadata[lookup_key] = ''

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
            for id, nuc_label in enumerate(exon_intron):
                print ('{},{},{}'.format(metadata['sequence_id'], id, nuc_label), file=fout)

