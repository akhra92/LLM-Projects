import torch
from transformers import BertTokenizer
from model import BertModelHandler


class Deployer:
    def __init__(self, model_handler, tokenizer, max_length=128, batch_size=16, label_map={0: 'Negative', 1: 'Positive'}):
        self.model = model_handler.get_model()
        self.device = model_handler.get_device()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.label_map = label_map

    def predict(self, sentences):
        encodings = self.tokenizer(
            sentences,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        dataset = torch.utils.data.TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask']
        )

        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size)

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids, attention_mask = [x.to(self.device) for x in batch]
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())

        results = []
        for review, pred in zip(sentences, all_preds):
            pred_label = self.label_map.get(pred, str(pred))
            results.append((review, pred_label))

        limiter = 100
        print('{:{width}}\t{}'.format('Review', 'Prediction', width=limiter))
        print('-'*(limiter + 15))
        for review, pred in results:
            truncated = review[:limiter] + ('...'if len(review) > limiter else '')
            print('{:{width}}\t{}'.format(truncated, pred, width=limiter))

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_handler = BertModelHandler()
    deployer = Deployer(model_handler=model_handler, tokenizer=tokenizer)
    sample_reviews = ['I love this movie a lot! It is very interesting.',
                      'I really did not like this drama. Terrible acting and boring content.',
                      'It was alright. Not great, not terrible-just an average watch.',
                      'The visuals and soundtrack were stunning. I would highly recommend this movie.',
                      'One of the worst films I have ever seen in my life. Total waste of time!',
                      'I do not understand the hype. Boring and way too long.',
                      'Average direction and not bad performance. Worth watching!',
                      'Surprisingly good! But the length is a bit long. However, I would recommend watching it.',
                      'There were a few moments, but overall it was disappointing',
                      'An emotional ride from beginning to end. Truly unforgettable!']
    
    deployer.predict(sentences=sample_reviews)