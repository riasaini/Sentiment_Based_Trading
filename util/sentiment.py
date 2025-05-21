import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
model.eval()

#Analyzes sentiment of a single paragraph
def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).squeeze().tolist()
    return dict(zip(['positive', 'negative', 'neutral'], probs))

# Extract summary sentiment features from a list of paragraphs
def get_sentiment_features(paragraphs):
    results = [analyze_sentiment(p) for p in paragraphs if len(p.strip()) > 10]

    if not results:
        return {
            'mean_positive': None,
            'mean_neutral': None,
            'mean_negative': None,
            'std_positive': None,
            'std_neutral': None,
            'std_negative': None,
            'polarity_score': None,
            'polarity_std': None,
            'max_positive': None,
            'min_negative': None,
            'num_paragraphs': 0
        }

    df = pd.DataFrame(results)

    return {
        'mean_positive': df['positive'].mean(),
        'mean_neutral': df['neutral'].mean(),
        'mean_negative': df['negative'].mean(),
        'std_positive': df['positive'].std(),
        'std_neutral': df['neutral'].std(),
        'std_negative': df['negative'].std(),
        'polarity_score': (df['positive'] - df['negative']).mean(),
        'polarity_std': (df['positive'] - df['negative']).std(),
        'max_positive': df['positive'].max(),
        'min_negative': df['negative'].min(),
        'num_paragraphs': len(df)
    }
