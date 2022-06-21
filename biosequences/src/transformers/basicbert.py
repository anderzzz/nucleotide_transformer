'''Bla bla

'''
from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer

class BertTextSentimentClassifier(nn.Module):
    '''Bla bla

    '''
    def __init__(self,
                 config=None,
                 pretrained=None,
                 n_class_labels=1
                 ):

        super(BertTextSentimentClassifier, self).__init__()

        if config is None and pretrained is None:
            raise ValueError('Both `config` and `pretrained` are `None`. One must be non-None.')

        self.bert = None
        if pretrained is None:
            if not isinstance(config, BertConfig):
                raise TypeError('BERT configuration not of required type {}'.format(type(BertConfig)))
            else:
                self.bert = BertModel(config=config)
        else:
            self.bert = BertModel.from_pretrained(pretrained)

        classifier_dropout = (
            self.bert.config.classifier_dropout if self.bert.config.classifier_dropout is not None
                                                else self.bert.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(in_features=self.bert.config.hidden_size,
                                    out_features=n_class_labels)

    def forward(self, text_tokens):
        output = self.bert(**text_tokens)
        class_pred = self.classifier(self.dropout(output.pooler_output))
        return class_pred

