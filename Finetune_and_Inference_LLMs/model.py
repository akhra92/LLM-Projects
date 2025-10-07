import torch
from transformers import BertForSequenceClassification
from torch.optim import AdamW


class BertModelHandler:
    def __init__(self, model_name='bert-base-uncased', num_labels=2, learning_rate=2e-5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
    
    def get_model(self):
        return self.model
    
    def get_optimizer(self):
        return self.optimizer
    
    def get_device(self):
        return self.device