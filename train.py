import os
import numpy as np
import torch
from torch import nn
from src import transformer
from torch import optim
from torch.utils.data import DataLoader, Dataset
from src.transformer import Transformer

def train(transformer_model: Transformer, data: Dataset, **kwargs):
    batch_size = kwargs.get("batch_size", 128)
    epoch = kwargs.get("epoch", 2)
    log_interval = kwargs.get("log_interval", 100)
    save_model_dir = kwargs.get("save_model_dir", './models/')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformer_model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer_model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    n_data = len(data_loader)

    transformer_model.train()
    
    for e in range(epoch):
        total_loss = 0.0
        for i, (src, tgt) in enumerate(data_loader):
            n_dots_for_print = ((i%3)+1)*'.'
            print(f"{(i/n_data)*100:>5.2f}% The model is Training{n_dots_for_print} ", end='\r')
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_x = tgt[:, :-1]
            tgt_y = tgt[:, 1:]

            y = transformer_model(src, tgt_x)
            y_dim = y.shape[-1]
            y = y.contiguous().view(-1, y_dim)
            tgt_y = tgt_y.contiguous().view(-1)

            optimizer.zero_grad()
            loss = criterion(y, tgt_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if i%log_interval==0 and i!=0:
                avg_loss = total_loss/log_interval
                print(f"Epoch: {e:^4} | Batch: {i/n_data:^6.3f} | Loss: {avg_loss:^6.4f}")

            # Save model
            if e%10==0 and e!=0:
                checkpoint_path = os.path.join(save_model_dir, f"transformer_epoch_{epoch}.pt")
                os.makedirs(save_model_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': transformer_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

train("", "")

