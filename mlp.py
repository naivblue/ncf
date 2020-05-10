# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPnet(nn.Module):
    def __init__(self, user_size, item_size, embedding_size, layers):
        super(MLPnet, self).__init__()
        self.user_encoder = nn.Embedding(user_size, embedding_size)
        self.item_encoder = nn.Embedding(item_size, embedding_size)
        self.decoder = nn.Linear(embedding_size, 1)
        self.logistic = torch.nn.Sigmoid()

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

    def forward(self, users, items):
        user_embeded = self.user_encoder(users)
        item_embeded = self.item_encoder(items)
        out = torch.cat([user_embeded, item_embeded], dim=-1)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            out = self.fc_layers[idx](out)
            out = F.relu(out)

        logits = self.decoder(out)
        rating = self.logistic(logits)
        return rating

