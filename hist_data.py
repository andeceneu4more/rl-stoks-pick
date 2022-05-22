import requests
from termcolor import colored as cl

def get_customized_news(stock, start_date, end_date, n_news, api_key, offset = 0):
    url = f'https://eodhistoricaldata.com/api/news?api_token={api_key}&s={stock}&limit={n_news}&offset={offset}&from={start_date}&to={end_date}'
    news_json = requests.get(url).json()
    
    news = []
    
    for i in range(len(news_json)):
        title = news_json[-i]['title']
        news.append(title)
        print(cl('{}. '.format(i+1), attrs = ['bold']), '{}'.format(title))
    
    return news

api_key= '628a3d1de04243.73158815'
appl_news = get_customized_news('AAPL', '2021-11-09', '2021-11-11', 15, api_key, 0)
print(appl_news)