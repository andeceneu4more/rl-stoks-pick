#!/usr/bin/env python

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
import pandas as pd
import torch
from os import listdir
from os.path import isfile, join

SEQ_LEN = 256

def get_sentiment(symbol, filename, max_news = 50): #format yyyy-mm-dd
    data = pd.read_csv(filename)
    for i, row in data.iterrows():
        date = row["Date"]

        tick = yf.Ticker(symbol) #"MSFT"
        tick = tick.history(start=date, end = date)
        results = tick.news[:max_news]

        #print(results)

        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        titles = [res["title"] for res in results]

        ids = tokenizer(titles, max_length=SEQ_LEN, truncation=True, return_tensors='pt', padding=True)

        outputs = model(**ids)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
        #print(predictions)

        positive = predictions[:, 0].tolist()
        negative = predictions[:, 1].tolist()
        neutral = predictions[:, 2].tolist()

        #table = {'Headline': titles, "Positive": positive, "Negative": negative, "Neutral": neutral}
        #table = {"Date":date,"Positive": positive, "Negative": negative}
        #df = pd.DataFrame(table, columns=["Positive", "Negative", "Neutral"]) #"Headline"
        df.at[i,'Positive'] = positive
        df.at[i,'Negative'] = negative

    name, extension = filename.split(".")
    data.to_csv(name + "WithSentiment." + extension, index=False)

for f in listdir("data"):
    if isfile(join("data", f):
        get_sentiment("AAPL",os.path.join("data",f))