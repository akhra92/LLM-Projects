from datasets import load_dataset
from transformers import BertTokenizer


class LoadDataset:
    def __init__(self, dataset_name='imdb'):
        self.dataset_name = dataset_name
        self.dataset = None

    def load(self):
        self.dataset = load_dataset(self.dataset_name)
        return self.dataset
    

class DataPreprocessor:
    def __init__(self, pretrained_model_name="bert-base-uncased", max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        self.max_length = max_length

    def preprocess(self,examples):
        return self.tokenizer(examples['text'],
                              padding='max_length',
                              max_length=self.max_length,
                              truncation=True,
                              return_tensors='pt')