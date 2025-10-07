import torch
from torch.utils.data import DataLoader
from dataset import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


class Inference:
    '''Inference with Bert Model for Sequence Semantic Classification
    Input: Model
    Output: Binary Predictions
    '''

    def __init__(self, model_handler, batch_size=16):
        self.model = model_handler.get_model()
        self.batch_size = batch_size
        self.device = model_handler.get_device()
    
    def predict(self, test_loader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].squeeze(1).to(self.device)
                attention_mask = batch['attention_mask'].squeeze(1).to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = logits.argmax(dim=-1)
                predictions.extend(preds.cpu().tolist())
        
        return predictions

    def preprocess(self, test_dataset):
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return test_loader
    


class SOTAInference:
    '''Inference with SOTA LLM Models'''
    def __init__(self, dataset_name, model_name, split):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.split = split

    def inference(self, prompt: str):
        '''Inference with the first example from the selected dataset using the selected model'''
        dataset = load_dataset(self.dataset_name, split=self.split)
        sentence = dataset[0]['text']

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)

        prompt = f"{prompt} this: {sentence}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = model.generate(**inputs, max_new_tokens=32)
        print("Prediction:", tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == '__main__':
    phi_model = SOTAInference(dataset_name='ag_news', model_name='microsoft/phi-3-mini-4k-instruct', split='test[:2]')
    phi_model.inference(prompt='Classify')