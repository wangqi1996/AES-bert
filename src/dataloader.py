# encoding=utf-8

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from src.util.util import get_filename, required_field


class EssayDataset(Dataset):

    def __init__(self, set_id, split, args):
        self.data = {}
        self.normalize_dict = {}
        self.feature = {}  # 存储feature和中间变量
        self.set_id = set_id
        self.filename = get_filename(split, args.data_dir)
        field_require = required_field(split)
        self.load_from_raw_file(self.filename, field_require)

    def load_from_raw_file(self, filename, field_require):

        data = pd.read_csv(filename, delimiter='\t')

        # Tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)
        for set_id in [self.set_id]:

            set_df = data[data.essay_set == set_id]
            fields = [set_df[field] for field in field_require]

            self.normalize_dict[str(set_id)] = {}
            self.data[str(set_id)] = []

            for values in tqdm(zip(*fields), total=len(fields[0])):
                sample_dict = {}
                for i, field in (enumerate(field_require)):
                    sample_dict[field] = values[i]

                self.data[str(set_id)].append(sample_dict)

    def __getitem__(self, item):
        # item: index
        essay = self.data[item]['essay']
        score = self.data[item]['score']

        return essay, score

    def __len__(self):
        return len(self.data)
