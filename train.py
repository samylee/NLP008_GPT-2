import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from models import GPT_2
from dataset import GPT2Dataset


def train(train_loader, model, criterion, optimizer, scheduler, device, epoch):
    model.train()
    losses = torch.zeros(len(train_loader))
    for i, (input, target) in enumerate(train_loader):
        cur_lr = optimizer.param_groups[0]['lr']

        # data to device
        input = input.to(device)
        target = target.to(device)

        # forward
        logits = model(input)

        # loss
        loss = criterion(logits.reshape(-1, logits.shape[-1]), target.view(-1))
        losses[i] = loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i%100 == 0:
            print("epoch: {}, iter: {}/{}, cur_lr: {:.6f}, train loss: {:.2f}".format(
                epoch + 1, i, len(train_loader), cur_lr, loss.item()))

    return torch.mean(losses)


def main():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    corpus_path = 'data/chinese_dialogue_instruction.parquet'

    batch_size = 32
    epochs = 100
    lr = 1e-3

    block_size = 512
    n_embed = 512
    n_heads = 8
    n_layers = 8
    dropout_ratio = 0.1

    checkpoint_dir = 'weights/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # load dataset
    print('Load dataset ... ')
    train_data = GPT2Dataset(corpus_path, block_size)
    data_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True)
    num_warmup_steps = int(len(data_loader) * epochs * 0.1)
    num_training_steps = int(len(data_loader) * epochs)

    # load model
    print('Load model ... ')
    model = GPT_2(train_data.tokenizer.vocab_size, block_size, n_embed, n_heads, n_layers, dropout_ratio)
    model = model.to(device)

    # loss and optimizer
    print('Load optimizer ... ')
    criterion = nn.CrossEntropyLoss(ignore_index=-1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9,0.999), weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    # training loop
    print('Start train loop ... ')
    for epoch in range(epochs):
        train_loss = train(data_loader, model, criterion, optimizer, scheduler, device, epoch)
        print("epoch: {}, train avg loss: {:.2f}".format(epoch + 1, train_loss))
        torch.save(model.state_dict(), checkpoint_dir + "epoch_{}_loss_{:.2f}.pt".format(epoch + 1, train_loss))


if __name__ == "__main__":
    main()
