# coding=utf-8
import os

import yaml


def get_filename(split, data_dir):
    mapping = {
        "train": "train.tsv",
        "dev": "dev.tsv",
        "test": "test.tsv"
    }
    name = mapping[split]
    return os.path.join(data_dir, name)


def required_field(split):
    mapping = {
        "train": ["essay_id", "essay_set", "essay", "domain1_score"],
        "dev": ["essay_id", "essay_set", "essay"],
        "test": ["essay_id", "essay_set", "essay"],
    }
    return mapping[split]


def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data = yaml.load(open(args.config_file))
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                arg_dict[key] = []
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


def get_label_num(set_id):
    mapping = {
        0: (0, 60),
        1: (2, 12),
        2: (1, 6),
        3: (0, 3),
        4: (0, 3),
        5: (0, 4),
        6: (0, 4),
        7: (0, 30),
        8: (0, 60)
    }
    result = mapping[set_id]
    return result[0], result[1], result[1] - result[0]
