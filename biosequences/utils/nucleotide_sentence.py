'''Bla bla

'''
class SequenceSentenceException(Exception):
    pass

class SequenceProcessException(Exception):
    pass

MAX_FAILURE = 1000

def _count_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)

def _nuc_checker(line):
    if ('N' in line) or ('n' in line):
        return False, 'Undefined nucleotide in sequence chunk'
    else:
        return True, ''

def make_sequence_sentences(fp_seq, fp_out, n_attempts,
                            lower_id_frac_generator,
                            sentence_length_generator,
                            nucleotide_checker,
                            lower_id_frac_generator_kwargs={},
                            sentence_length_generator_kwargs={},
                            nucleotide_checker_kwargs={}):
    '''Bla bla

    '''
    #
    # Efficient determination of the number of lines in large sequence source file
    with open(fp_seq, 'rb') as fp:
        c_generator = _count_generator(fp.raw.read)
        count = sum(buffer.count(b'\n') for buffer in c_generator)
        n_lines_seq = count + 1

    #
    # Header to output file
    print('ind,sequence_length,start_line_index,sequence_chunk', file=fp_out)

    #
    # Create array of indices.
    file_indices = [(lower_id_frac_generator(**lower_id_frac_generator_kwargs),
                     lower_id_frac_generator(**lower_id_frac_generator_kwargs),
                     sentence_length_generator(**sentence_length_generator_kwargs)) for _ in range(n_attempts)]
    file_indices = sorted(file_indices, key=lambda x: x[0])

    with open(fp_seq) as fin:

        append_to_collection = False
        k_attempt = 0
        collection = []
        for k_line, line in enumerate(fin):
            if append_to_collection:
                ok, remark = nucleotide_checker(line, **nucleotide_checker_kwargs)
                if ok:
                    collection.append(line.replace('\n',''))
                    sentence_len = file_indices[k_attempt][2]
                    if sum([len(x) for x in collection]) >= sentence_len:
                        sentence = ''.join(collection)
                        sentence = sentence[:sentence_len]
                        print ('{},{},{},{}'.format(k_attempt,sentence_len, k_line_start, sentence), file=fp_out)
                        print ('row added: {}'.format(k_attempt))

                        collection = []
                        append_to_collection = False

                else:
                    collection = []
                    append_to_collection = False

            else:
                row_mask = [k_line == int(n_lines_seq * f[0]) for f in file_indices]
                if any(row_mask):
                    k_line_start = k_line
                    k_attempt = row_mask.index(True)
                    line_ = line.replace('\n','')
                    ok, remark = nucleotide_checker(line_, **nucleotide_checker_kwargs)
                    if ok:
                        col_ind = int(len(line_) * file_indices[k_attempt][1])
                        collection.append(line_[col_ind:])
                        append_to_collection = True

                    else:
                        collection = []
                        append_to_collection = False


from numpy.random import default_rng
rng = default_rng()
with open('/Users/andersohrn/PycharmProjects/nucleotide_transformer/test.csv', 'w') as fout:
    make_sequence_sentences(fp_seq='/Users/andersohrn/PycharmProjects/nucleotide_transformer/ncbi-genomes-2022-07-05/GCF_000001405.40_GRCh38.p14_genomic.fna',
                            fp_out=fout,
                            n_attempts=100,
                            lower_id_frac_generator=rng.random,
                            sentence_length_generator=rng.integers,
                            sentence_length_generator_kwargs={'low' : 10, 'high' : 1000},
                            nucleotide_checker=_nuc_checker)