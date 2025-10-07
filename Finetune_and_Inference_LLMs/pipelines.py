from transformers import pipeline
import argparse

class Pipelines:
    def __init__(self, args):
        self.args = args

    def sentiment_analysis(self):
        sentiment_analysis = pipeline('sentiment-analysis', device=self.args.device, model='distilbert-base-uncased-finetuned-sst-2-english')
        result = sentiment_analysis(self.args.input)[0]
        print(f"AI Decision:\n\nSentence\t\t-> {self.args.input}\nSentiment\t\t-> {result['label']}")

    def translation(self):
        translator = pipeline('translation', model='acebook/nllb-200-distilled-600M', device=self.args.device)
        result = translator(self.args.input, src_lang=self.args.src_lang, tgt_lang=self.args.tgt_lang)
        print(f"Sentence\t\t-> {self.args.input}")
        print(f"Translation to {self.args.tgt_lang}\t\t-> {result[0]['translation_text']}")

    def image_classification(self, image):
        img_classifier = pipeline('image-classification', device=self.args.device, use_fast=True)
        cls_results = img_classifier(image)
        print(f"Image Classified as\t\t->{cls_results[0]['label']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='Yusuf kaltak yeydigan bola ekan.', help='Enter a sentence for processing!')
    parser.add_argument('--device', type=str, default='mps', help='Enter device (cpu, cuda, mps)')
    parser.add_argument('--src_lang', type=str, default='eng_Latn', help='Enter a source language for translation task')
    parser.add_argument('--tgt_lang', type=str, default='kor_Hang', help='Enter a target language for translation task')

    args = parser.parse_args()

    processor = Pipelines(args = args)
    processor.translation()

