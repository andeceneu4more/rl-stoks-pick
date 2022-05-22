#!/usr/bin/env python

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import yfinance as yf
import pandas as pd
import torch
import os
from os.path import isfile, join
from datetime import datetime

SEQ_LEN = 256


def get_sentiment(symbol, filename, max_news=50):  # format yyyy-mm-dd
    data = pd.read_csv(filename)

    tick = yf.Ticker(symbol)  # "MSFT"
    #tick = tick.history(start=date, end=date)
    results = tick.news#[:max_news]
    results['providerPublishTime'] = datetime.fromtimestamp(results['providerPublishTime']).date()
    titles_by_date = dict()
    for res in results:
        if res['providerPublishTime'] in res:
            titles_by_date[res['providerPublishTime']].append(res['title'])
        else:
            titles_by_date[res['providerPublishTime']] = [res['title']]

    print(titles_by_date)
    for i, row in data.iterrows():
        date = row["Date"]

        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

        titles = [res["title"] for res in titles_by_date[date]][:max_news]
        ids = tokenizer(titles, max_length=SEQ_LEN, truncation=True, return_tensors='pt', padding=True)

        outputs = model(**ids)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
        # print(predictions)

        positive = predictions[:, 0].tolist()
        positive = float(sum(positive) / len(positive))
        negative = predictions[:, 1].tolist()
        negative = float(sum(negative) / len(negative))
        # neutral = predictions[:, 2].tolist()

        data.at[i, 'Positive'] = positive
        data.at[i, 'Negative'] = negative

    name, extension = filename.split(".")
    data.to_csv(name + "WithSentiment." + extension, index=False)


for f in os.listdir("data"):
    if isfile(join("data", f)):
        get_sentiment(f.split("_")[0], os.path.join("data", f))
