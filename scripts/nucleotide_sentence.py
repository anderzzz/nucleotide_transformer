'''Functions to generate nucleotide sentences from a large source file of nucleotide sequences.

A typical example is to generate sentences from random spots along a full genome, such as the
fasta file of the Genome Reference Consortium Human Build 38 patch release 14 (GRCh38.p14), see
https://www.ncbi.nlm.nih.gov/assembly/GCF_000001405.40.

'''
class SequenceSentenceException(Exception):
    pass

class SequenceProcessException(Exception):
    pass

def _count_generator(reader):
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024 * 1024)

def _nuc_checker(line):
    '''Criteria to approve/deny a sequence for inclusion. Typically deny unknown nucleotides, labelled N or n'''
    if ('N' in line) or ('n' in line):
        return False, 'Undefined nucleotide in sequence chunk'
    else:
        return True, ''

def make_sequence_sentences(fp_seq,
                            fp_out, out_ind_offset,
                            n_attempts,
                            lower_id_frac_generator,
                            sentence_length_generator,
                            nucleotide_checker,
                            lower_id_frac_generator_kwargs={},
                            sentence_length_generator_kwargs={},
                            nucleotide_checker_kwargs={}):
    '''Main function to generate sentences.

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
        sentence_target_len = None
        sentence_build = None
        sentence_build_len = 0

        #
        # Iterate one line at a time from the file
        for k_line, line in enumerate(fin):

            #
            # If line appending is active, go here
            if append_to_collection:
                ok, remark = nucleotide_checker(line, **nucleotide_checker_kwargs)
                if ok:
                    line_ = line[:-1]
                    sentence_build += line_
                    sentence_build_len += len(line_)

                    #
                    # If enough number of characters have been added to sentence, print to disk
                    if sentence_build_len >= sentence_target_len:
                        sentence = sentence_build[:sentence_target_len]
                        print ('{},{},{},{}'.format(k_attempt,
                                                    sentence_target_len,
                                                    k_line_start + out_ind_offset,
                                                    sentence),
                               file=fp_out)
                        print ('row added: {}'.format(k_attempt))

                        sentence_build = None
                        append_to_collection = False

                #
                # If invalid nucleotide encountered, stop all further aggregation
                else:
                    sentence_build = None
                    append_to_collection = False

            #
            # Check if this line should initiate sentence creation
            else:
                #
                # The row mask tests if current line corresponds to one of the random file fractions
                row_mask = [k_line == int(n_lines_seq * f[0]) for f in file_indices]
                if any(row_mask):
                    ok, remark = nucleotide_checker(line, **nucleotide_checker_kwargs)
                    if ok:
                        k_line_start = k_line
                        k_attempt = row_mask.index(True)
                        line_ = line[:-1]
                        col_ind = int(len(line_) * file_indices[k_attempt][1])
                        sentence_target_len = file_indices[k_attempt][2]
                        line_ = line_[col_ind:]
                        sentence_build = line_
                        sentence_build_len = len(line_)

                        append_to_collection = True

                    else:
                        sentence_build = None
                        append_to_collection = False


if __name__ == '__main__':
    from numpy.random import default_rng
    rng = default_rng()
    with open('/test_xak.csv', 'w') as fout:
        make_sequence_sentences(fp_seq='/Users/andersohrn/PycharmProjects/nucleotide_transformer/ncbi-genomes-2022-07-05/xak',
                                fp_out=fout,
                                out_ind_offset=39999999,
                                n_attempts=308,
                                lower_id_frac_generator=rng.random,
                                sentence_length_generator=rng.integers,
                                sentence_length_generator_kwargs={'low' : 5, 'high' : 1000},
                                nucleotide_checker=_nuc_checker)