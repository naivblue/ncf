#!/usr/bin/env python
#!coding=utf8

import os
import torch
import torch.nn as nn
import torch.optim as optim

from evaluation import evaluation
from data_utils import load_data
from mlp import MLPnet
from data import Dataset
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--epochs', type=int, default=100, help='the number of maximum training epoch')
parser.add_argument('--batch_size', type=int, default=1000, help='size of batch in mini-batch training')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate of optimizer')
parser.add_argument('--embedding_size', type=int, default=8, help='dimension of embedding')
parser.add_argument('--train_num_negative', type=int, default=4, help='negative sampling num for train ')
parser.add_argument('--test_num_negative', type=int, default=99, help='negative sampling num for test ')
parser.add_argument('--layer', type=list, default=[16, 64, 32, 16, 8], help='layers in MPL model ')
#parser.add_argument('--model_path', type=str, default='./log/', help='path to save trained model')
parser.add_argument('--use_cuda', action='store_true', help='use cuda')
args = parser.parse_args()


def train_batch(users, items, targets, model, optimizer, criterion):
    if args.use_cuda is True:
        users, items, targets = users.cuda(), items.cuda(), targets.cuda()
    optimizer.zero_grad()
    output = model(users, items)
    loss = criterion(output.view(-1), targets)
    loss.backward()
    optimizer.step()
    loss = loss.item()
    return loss


def train_epoch(train_data, model, optimizer, criterion):
    model.train()
    total_loss = 0
    for batch_id, batch in enumerate(train_data):
        users, items, targets = batch[0], batch[1], batch[2]
        loss = train_batch(users, items, targets, model, optimizer, criterion )
        total_loss += loss


def main():
    # Load data
    data = load_data()
    dataset = Dataset(ratings=data)
    evaluation_data = dataset.test_data_loader(args.test_num_negative)

    # Build model
    model = MLPnet(user_size=dataset.user_size, item_size=dataset.item_size, embedding_size=args.embedding_size, layers=args.layer)
    if args.use_cuda is True:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Train model
    print('start train...')
    for epoch in range(args.epochs):
        train_data = dataset.train_data_loader(args.train_num_negative, args.batch_size)
        train_epoch(train_data, model, optimizer, criterion)

        # evaluation per train epoch
        evaluation(evaluation_data, args.use_cuda, model)

        # save model per train epoch
        if epoch % 10 == 0:
            #if not os.path.exists(args.model_path):
            #    os.mkdir(args.model_path)
            torch.save(model.state_dict(), '/model.pt')
            print("[Saved the trained model successfully.]")


if __name__ == '__main__':
    main()