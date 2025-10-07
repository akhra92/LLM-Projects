from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model_handler, trn_loader, val_loader, batch_size=16):
        self.model = model_handler.get_model()
        self.optimizer = model_handler.get_optimizer()
        self.device = model_handler.get_device()
        self.batch_size = batch_size
        self.trn_loader = trn_loader
        self.val_loader = val_loader

    def train_epoch(self):
        
        self.model.train()    

        total_loss = correct = total = 0

        for batch in tqdm(self.trn_loader, desc='Training...'):
            input_ids = batch['input_ids'].squeeze(1).to(self.device)
            attention_mask = batch['attention_mask'].squeeze(1).to(self.device)
            labels = batch['labels'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(self.trn_loader)
        accuracy = correct / total

        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = correct = total = 0

        for batch in tqdm(self.val_loader, desc='Validating...'):
            input_ids = batch['input_ids'].squeeze().to(self.device)
            attention_mask = batch['attention_mask'].squeeze().to(self.device)
            labels = batch['labels'].to(self.device)

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total

        return avg_loss, accuracy
