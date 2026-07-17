from transformers import AutoModelForSequenceClassification, AutoTokenizer
model_name = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained(model_name)
print('Downloading model...')
AutoModelForSequenceClassification.from_pretrained(model_name)
print('Done. Cached in ~/.cache/huggingface/')

from transformers import pipeline
import torch
device = 0 if torch.cuda.is_available() else -1
print(f'CUDA available: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()}')
pipe = pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest',
device=device)
results = pipe(
    ['This is really helpful, thank you!', 'I dont think that a good idea at all.'], 
    batch_size=2, 
    truncation=True
)
for text, r in zip(['helpful', 'not good idea'], results):
    print(f'  {text}: {r}')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
print(f'VADER fallback OK: {analyzer.polarity_scores("test")}')