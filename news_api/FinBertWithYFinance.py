#!/usr/bin/env python

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
import pandas as pd
import torch

SEQ_LEN = 256
MAX_NEWS = 50
msft = yf.Ticker("MSFT")
results = msft.news[:MAX_NEWS]

print(results)

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

titles = [res["title"] for res in results]

ids = tokenizer(titles, max_length=SEQ_LEN, truncation=True, return_tensors='pt', padding=True)

outputs = model(**ids)
predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
print(predictions)

positive = predictions[:, 0].tolist()
negative = predictions[:, 1].tolist()
neutral = predictions[:, 2].tolist()

#table = {'Headline': titles, "Positive": positive, "Negative": negative, "Neutral": neutral}
table = {"Positive": positive, "Negative": negative, "Neutral": neutral}
df = pd.DataFrame(table, columns=["Positive", "Negative", "Neutral"]) #"Headline"

print(df.head(5))
