#!/usr/bin/env python

from urllib.request import urlopen
import certifi
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def get_jsonparsed_data(url):
    response = urlopen(url, cafile=certifi.where())
    data = response.read().decode("utf-8")
    return json.loads(data)

SEQ_LEN = 256
YOUR_API_KEY="36d3801ed2b3e85fd3b3755681e3e416"

url = "https://financialmodelingprep.com/api/v3/stock_news?tickers=AAPL&limit=100&apikey=" + YOUR_API_KEY
results = get_jsonparsed_data(url)
print(results)

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

for res in results:
    text = res["text"]
    ids, segments = tokenizer.encode(text, max_len=SEQ_LEN)
    inpu = np.array(ids).reshape([1, SEQ_LEN])
    predicted_id = model.predict([inpu,np.zeros_like(inpu)]).argmax(axis=-1)[0]
    print ("%s: %s"% predicted_id, text)