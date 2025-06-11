import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from src.transformer import Transformer
from data.translation_data import train_loader, valid_loader, src_vocab, tgt_vocab

model = Transformer(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=512,
    num_heads=8,
    num_layers=6,
    d_ff=2048,
    max_seq_length=512,
    dropout=0.1
)

def train(transformer_model: Transformer, train_loader: DataLoader, valid_loader: DataLoader, **kwargs):
    epoch = kwargs.get("epoch", 2)
    log_interval = kwargs.get("log_interval", 100)
    save_model_dir = kwargs.get("save_model_dir", './models/')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    transformer_model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(transformer_model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    n_data = len(train_loader)

    transformer_model.train()
    
    for e in range(epoch):
        total_loss = 0.0
        for i, (src, tgt) in enumerate(train_loader):
            transformer_model.train()
            n_dots_for_print = ((i%3)+1)*'.'
            print(f"\033[96m{(i/n_data)*100:>5.2f}% The model is Training{n_dots_for_print} \033[00m", end='\r')
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

        loss_v = 0.0
        avg_loss = total_loss/log_interval

        # valid data
        transformer_model.eval()
        for src_v, tgt_v in valid_loader:
            src_v, tgt_v = src_v.to(device), tgt_v.to(device)
            tgt_v_x, tgt_v_y = tgt_v[:, :-1], tgt_v[:, 1:]
            y_v = transformer_model(src_v, tgt_v_x)
            y_v = y_v.contiguous().view(-1, y_v.shape[-1])
            tgt_v_y = tgt_v_y.contiguous().view(-1)
            loss_v += criterion(y_v, tgt_v_y).item()
        avg_loss_v = loss_v/log_interval

        print(f"Epoch: {e:^4} | Loss: {avg_loss:^6.4f} | valid loss: {avg_loss_v:^6.4f}")
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

if __name__=="__main__":
    train(model, train_loader, valid_loader)
