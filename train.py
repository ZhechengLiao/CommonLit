import torch
from model import BERT_Model
from dataset import *


def train(num_epochs, loss_fn, optimizer, device):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for step, (input_ids, attention_mask, content, wording) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            content = content.to(device)
            wording = wording.to(device)

            output = model(input_ids, attention_mask)
            loss = loss_fn(output[:, 0], content) + loss_fn(output[:, 1], wording)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print(f'Epoch: {epoch+1}, Step: {step}, Loss: {loss.item()}')
            running_loss += loss.item()
        print(f'Epoch: {epoch+1}, Loss: {running_loss/len(train_loader)}')
        
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for val_step, (input_ids, attention_mask, content, wording) in enumerate(val_loader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                content = content.to(device)
                wording = wording.to(device)

                output = model(input_ids, attention_mask)
                loss = loss_fn(output[:, 0], content) + loss_fn(output[:, 1], wording)
                val_loss += loss.item()
            
        print(f'Epoch: {epoch+1}, Val_Loss: {val_loss/len(val_loader)}')
                

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BERT_Model().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    train(30, loss_fn, optimizer, device)