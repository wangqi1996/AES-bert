# coding=utf-8
import nltk
import torch.nn as nn
from transformers import BertModel, BertTokenizer

MODELS = [(BertModel, BertTokenizer, 'bert-base-uncased')]


class BCNN(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.max_sentence_length = args.max_sentence_length
        if self.max_sentence_length > 510:
            self.max_sentence_length = 510

        self.pretrained_weights = args.model_name

        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_weights)
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token])[0]
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.sep_token])[0]

        self.bert = BertModel.from_pretrained(self.pretrained_weights).to(args.device)

        # self.classifier = CNN(args)
        self.gru = nn.GRU(input_size=args.bert_hidden_size, hidden_size=args.bert_hidden_size, bidirectional=True,
                          batch_first=True, dropout=args.gru_dropout)

        self.dropout = nn.Dropout(args.linear_dropout)

        self.classifier = nn.Linear(in_features=args.bert_hidden_size * 2, out_features=args.label_num)
        self.max_score = args.max_score
        self.min_score = args.min_score

        self.init_weights()

    def forward(self, essay):
        # data对应一个文章集合，多次调用bert，获取到整个文章的表示，然后输入分类器

        batch_tokens, longest_sentence_length = self.encode(essay)
        for _tokens in batch_tokens:
            pad_seq_length = min(longest_sentence_length, self.max_sentence_length)
            tokens = [self.cls_token_id] + _tokens[:pad_seq_length] + [self.sep_token_id]

            sentence_length = len(tokens)
            pad_seq_length += 2

            token_type_ids = [0] * sentence_length
            input_mask = [1] * sentence_length

            padding = [0] * (pad_seq_length - sentence_length)
            tokens += padding
            token_type_ids += padding
            input_mask += padding

    def encode(self, data):
        # 编码句子表示, 暂时一个句子一个句子的处理
        sentence_list = [[self.process_text(d)] for d in data]

        batch_tokens = []
        longest_sentence_length = 0
        for sentence in sentence_list:
            tokenized_text = self.tokenizer.encoder(sentence)
            batch_tokens.append(tokenized_text)
            longest_sentence_length = max(longest_sentence_length, len(tokenized_text))

        return batch_tokens, longest_sentence_length

    def process_text(self, essay):
        """ 按照句子分割 """
        sentences = nltk.sent_tokenize(essay)
        return sentences
