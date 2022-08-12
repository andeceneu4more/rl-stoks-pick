#!/usr/bin/env python

from urllib.request import urlopen
import certifi
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

SEQ_LEN = 256
YOUR_API_KEY="36d3801ed2b3e85fd3b3755681e3e416"

url = "https://financialmodelingprep.com/api/v3/stock_news?tickers=AAPL&limit=1&apikey=" + YOUR_API_KEY
results = get_jsonparsed_data(url)
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
