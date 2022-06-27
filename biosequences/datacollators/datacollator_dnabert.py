'''Bla bla

'''
import torch
from transformers import DataCollatorForLanguageModeling

class DataCollatorDNAWithMasking(DataCollatorForLanguageModeling):
    '''Bla bla

    '''
    def __init__(self, tokenizer, mlm_probability=0.15, pad_to_multiple_of=None, word_length=3):
        mlm_probability_parent = mlm_probability / word_length
        super(DataCollatorDNAWithMasking, self).__init__(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=mlm_probability_parent,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors='pt'
        )
        self.word_length = word_length

    def __call__(self, features, return_tensors='pt'):
        '''Bla bla

        '''
        if return_tensors != 'pt':
            raise NotImplementedError('DNA Datacollator only implemented for Pytorch. Unknown option: {}'.format(return_tensors))

        return super().torch_call(features)

    def shift_mask(self, t, direction, nsteps, impassable=None):
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

        ###RECURSION TO ROLL MASK see torch.roll

    def torch_mask_tokens(self, inputs, special_tokens_mask=None):
        '''Override parent method to handle the special issues with DNA masking

        '''
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        masked_indices = self.shift_mask(t=masked_indices,
                                         direction='left', nsteps=self.word_length // 2,
                                         impassable=special_tokens_mask)
        masked_indices = self.shift_mask(t=masked_indices,
                                         direction='right', nsteps=self.word_length // 2,
                                         impassable=special_tokens_mask)
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        return inputs, labels

