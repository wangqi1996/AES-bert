# coding=utf-8
import nltk
import torch.nn as nn
from transformers import *

MODELS = [(BertModel, BertTokenizer, 'bert-base-uncased')]


class BCNN(nn.Module):
    def __init__(self, args):
        self.pretrained_weights = 'bert-base-uncased'

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)
        self.bert = BertModel.from_pretrained(self.pretrained_weights).to(args.device)

        # self.classifier = CNN(args)
        self.gru = nn.GRU(input_size=args.bert_hidden_size, hidden_size=args.bert_hidden_size, bidirectional=True,
                          batch_first=True, dropout=args.gru_dropout)

        self.dropout = nn.Dropout(args.linear_dropout)

        self.classifier = nn.Linear(in_features=args.bert_hidden_size * 2, out_features=args.label_num)
        self.max_score = args.max_score
        self.min_score = args.min_score

        self.init_weights()

    def forward(self, data):
        # data对应一个文章集合，多次调用bert，获取到整个文章的表示，然后输入分类器
        # data: [batch_size]
        sentence_list = [[self.process_text(d)] for d in data]

        for sentence in sentence_list:
            tokenized_text = self.tokenizer.encoder()

    def process_text(self, essay):
        """ 按照句子分割 """
        sentences = nltk.sent_tokenize(essay)
        return sentences
