# coding=utf-8
from argparse import ArgumentParser

import torch
from torch.utils.data.dataloader import DataLoader

from src.dataloader import EssayDataset
from src.model.bcnn import BCNN
from src.util.util import parse_args, get_label_num


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    args.min_score, args.max_score, args.label_num = get_label_num(args.set_id)

    train_dataset = EssayDataset(args.set_id, 'train', args)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0)

    dev_dataset = EssayDataset(args.set_id, 'dev', args)

    dev_loader = DataLoader(dev_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0)

    model = BCNN(args)

    for batch in train_loader:
        print("batch_1")
        print(batch)


if __name__ == '__main__':
    args_parser = ArgumentParser()

    args_parser.add_argument('--config_file', '-c', default='config.yml', type=str)
    args_parser.add_argument('--train', action="store_true", default=False)
    args_parser.add_argument('--set_id', type=int, default=1)  # 1-8
    args = parse_args(args_parser)

    main(args)
