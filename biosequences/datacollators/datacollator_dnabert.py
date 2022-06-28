'''Bla bla

'''
import torch
from transformers import DataCollatorForLanguageModeling

class DataCollatorDNAWithMasking(DataCollatorForLanguageModeling):
    '''Custom data collator with masking for DNA tokens.

    '''
    def __init__(self, tokenizer, mlm_probability=0.15, pad_to_multiple_of=None,
                 word_length=3, allow_mask_at_edge=False):
        mlm_probability_parent = mlm_probability / word_length
        super(DataCollatorDNAWithMasking, self).__init__(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability_parent,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors='pt'
        )
        self.word_length = word_length
        self.allow_mask_at_edge = allow_mask_at_edge

    def __call__(self, features, return_tensors='pt'):
        '''Wraps the parent Pytorch method that a call to the data collator invokes. The parent method in turn
        invokes a method for masking that is overridden in a method of the present class

        '''
        if return_tensors != 'pt':
            raise NotImplementedError('DNA Datacollator only implemented for Pytorch. Unknown option: {}'.format(return_tensors))

        return super().torch_call(features)

    def shift_mask(self, t, direction, nsteps, impassable=None):
        '''Recursive shifting of masking tensor set number of steps, left or right

        The method further ensures that shifting the mask does not: (1) Cross the edge to the other side. So shifting
        left will not suddenly create masking at the rightmost edge (2) Cross any impassable tokens. These are
        going to be special tokens. So if the last token in a sequence is masked, the mask will not expand to the
        first token in the next sequence in the batch.
        '''
        if nsteps == 0:
            return t
        if direction == 'left':
            move = -1
            edge = -1
        elif direction == 'right':
            move = 1
            edge = 0
        else:
            raise ValueError('The direction must be either `left` or `right`, not {}'.format(direction))

        t_shift = torch.roll(t, shifts=move)
        t_shift[:, edge] = False

        t_union = (t | t_shift) & ~impassable

        return self.shift_mask(t=t_union, direction=direction, nsteps=nsteps - 1, impassable=impassable)

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        '''Custom method to mask DNA tokens. This overrides the parent method, which only masks one token,
        which is an insufficient approach in case of DNA token vocabulary with word length greater than one.

        '''
        labels = inputs.clone()

        #
        # Sample tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        #
        # Special tokens should never be masked. Also, since masked DNA codons at the edge of the
        # batch cannot be shifted to neighbouring batches, it is better to disallow their masking
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        if not self.allow_mask_at_edge:
            probability_matrix[:, :self.word_length // 2] = 0.0
            probability_matrix[:, -1 * self.word_length // 2:] = 0.0

        #
        # Generate random masking tensor. Then expand the mask left and right, such that it
        # covers all DNA tokens that contain the central nucleotide of the mask
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = self.shift_mask(t=masked_indices,
                                         direction='left', nsteps=self.word_length // 2,
                                         impassable=special_tokens_mask)
        masked_indices = self.shift_mask(t=masked_indices,
                                         direction='right', nsteps=self.word_length // 2,
                                         impassable=special_tokens_mask)

        #
        # Setting labels to -100 means that the loss computation will disregard prediction of
        # all non-masked tokens
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels

