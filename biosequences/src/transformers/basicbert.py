'''Bla bla

'''
from torch import nn
from transformers import BertModel, BertConfig, BertTokenizer

class BertTextSentimentClassifier(nn.Module):
    '''Bla bla

    '''
    def __init__(self,
                 config=None,
                 pretrained='bert-base-uncased',
                 n_class_labels=1
                 ):

        super(BertTextSentimentClassifier, self).__init__()

        self.bert = None
        if pretrained is None:
            self.bert = BertModel(config=config)
        else:
            if not config is None:
                raise RuntimeWarning('Bert configuration ignored when pretrained parameter provided')
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
        pooled_output = output.pooler_output
        pooled_output = self.dropout(pooled_output)
        class_pred = self.classifier(pooled_output)

        return class_pred

def create_bert_objs(config=None,
                     pretrained='bert-base-uncased'):
    '''Convenience function to create the bert model and associated tokenizer

    '''
    model = BertTextSentimentClassifier(config=config,
                                        pretrained=pretrained)
    if pretrained is None:
        pass
    else:
        tokenizer = BertTokenizer.from_pretrained(pretrained)

    return model, tokenizer

#config = BertConfig()
#model = BertModel.from_pretrained('bert-base-uncased')
#
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
#dummy = 'Well that was quite remarkable, quite remarkable indeed. Now tell me, dude, what is next?'
#inputs = tokenizer(dummy, return_tensors='pt')
#
#outputs = model(**inputs)
#
#print (outputs)
