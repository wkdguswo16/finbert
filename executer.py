MODEL_NAME = 'Compute'

import os
import ast
import torch
import numpy as np
import random as rn
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from yfinance import Ticker
import json
from typing import Tuple, List
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import env
import openai
openai.api_key = env.API_KEY
# Core data paths
DATE = datetime.now().strftime("%y%m%d")
MODEL_PATH = 'ipuneetrathore/bert-base-cased-finetuned-finBERT'
TICKER_PATH = 'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv'
NEWS_DATA_PATH = f'news_data_{DATE}.csv'
FINANCIAL_DATA_PATH = f'finencial_data_{DATE}.csv'
SUB_PATH = 'finbert_submission.csv'
financial_data = set(['TSLA', 'AAPL', 'DIS', 'GE', 'BRK-B', 'NKE', 'SBUX', 'SSNLF', 'AMZN', 'META', 'GOOGL', 'BRK-A', 'MSFT', 'TSM',
                       'UNH', 'JNJ', 'WMT', 'NVDA', 'PG', 'AMD', 'JPM', 'XOM', 'MA', 'CVX', 'HD', 'BAC', 'PFE', 'LLY', 'ABBY', 'KO', 'NVO', 'BABA', 'PEP', 'TM', 'ASML', 'COST', 'VZ', 'TMO', 'MRK', 'ORCL', 'AVGO', 'ABT', 'NVS', 'ACN', 'DHR', 'BHP', 'AZN', 'MCD', 'CRM', 'WFC', 'CSCO', 'BMY', 'UPS', 'LIN', 'QCOM', 'LMT', 'NFLX', 'ADBE', 'INTC'])


class Device:
    CPU = 'cpu'
    CUDA = 'cuda'
    MPS = 'mps'

    def __init__(self, target_device:str=None):
        if target_device is not None:
            self.device = target_device
            return
        self.device = Device.CPU
        if torch.cuda.is_available():
            self.device = Device.CUDA
        elif torch.backends.mps.is_available():
            self.device = Device.MPS
    
    def __repr__(self) -> str:
        return self.device

    def get_name(self) -> str:
        return self.device
    
class FinBertCased:
    def __init__(self, max_len: int, batch_size: int, model_path: str, device: Device):
        self.max_len = max_len
        self.model_path = model_path
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.device = device.get_name()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path).eval().to(self.device)
        self.label_dict = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.inverse_label_dict = {v: k for k, v in self.label_dict.items()}

    def full_preprocess(self, text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Preprocessing pipeline from string to ids and attention mask. """
        encoded = self.tokenizer(text,
                                 add_special_tokens=True,
                                 max_length=self.max_len,
                                 padding='max_length',
                                 return_attention_mask=True,
                                 return_tensors='pt',
                                 truncation=True)
        input_ids = torch.cat([encoded['input_ids']], dim=0).to(self.device)
        attention_mask = torch.cat(
            [encoded['attention_mask']], dim=0).to(self.device)
        return input_ids, attention_mask

    def predict_raw(self, text: List[str]) -> torch.Tensor:
        """ Predict raw logits """
        input_ids, attention_mask = self.full_preprocess(text)
        model_output = self.model(
            input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits = model_output[0]
        return logits

    def predict_score(self, text: List[str]) -> np.float32:
        """ Predict a single sentiment score (positive_sentiment - negative_sentiment) """
        logits = self.predict_raw(text)
        softmax_output = F.softmax(logits, dim=1).cpu().detach().numpy()
        pos_idx = self.inverse_label_dict['positive']
        neg_idx = self.inverse_label_dict['negative']
        return softmax_output[:, pos_idx] - softmax_output[:, neg_idx]

    def predict_signals(self, text: pd.Series) -> List[float]:
        """
        Get ranking of average sentiment scores for every ticker in the data.
        :param text: Pandas Series of articles grouped by week and ticker
        :return: Scaled sentiment scores in range [0...1]
        """
        sent_scores = []
        sent_length = []
        for row in tqdm(text):
            sents = row.split(" [SEP] ")[:-1]
            sent_length.append(len(sents))
            sent_scores_ticker = []
            for batch in self._chunks(sents, self.batch_size):
                batch_sents = self.predict_score(batch)
                sent_scores_ticker.append(batch_sents)
            mean_score = np.array(np.concatenate(
                sent_scores_ticker)).ravel().mean()
            sent_scores.append(mean_score)
        signals = self._scale_sentiment(sent_scores)
        return signals, sent_length

    @staticmethod
    def _chunks(lst, n):
        """ Yield successive n-sized chunks from list. """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    @staticmethod
    def _scale_sentiment(sentiments: List[float]):
        """ Scale sentiment scores from [-1...1] to [0...1] """
        mm = MinMaxScaler()
        sent_proc = np.array(sentiments).reshape(-1, 1)
        return mm.fit_transform(sent_proc)



class StockNewsProcessor:
    '''
    Preprocessor for Stock News API output
    https://stocknewsapi.com
    '''

    def __init__(self, financial_data:pd.DataFrame, unnecessary_cols=['uuid', 'link',
                                                                'publisher', 'thumbnail', 'type']):
        self.tickers = pd.read_csv(TICKER_PATH)
        self.unnecessary_cols = unnecessary_cols
        self.text_cols = ['title']
        self.pre_data = financial_data
        self.relevant_tickers = financial_data.ticker

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Data cleaning for Stock News API data

        :param data: A raw Pandas DataFrame containing at least the columns 'type', 'sentiment', 'providerPublishTime' and
                    columns specified in self.unnecessary_cols.
        :return: A clean and sorted DataFrame with date as index.
        '''
        # data['providerPublishTime'] = data['providerPublishTime'].apply(datetime.fromtimestamp)
        data = data.drop(self.unnecessary_cols, axis=1)
        data = data.drop_duplicates()
        data['relatedTickers'] = data['relatedTickers'].apply(
            lambda s: list(ast.literal_eval(s)))
        data['providerPublishTime'] = data['providerPublishTime'].apply(
            lambda x: datetime.fromtimestamp(x))
        data = data.set_index(data['providerPublishTime'], drop=True).sort_index()
        return data

    def aggregate(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Combine weekly text and order by Friday date

        :param data: A preprocessed DataFrame
        :return: Data grouped by ticker and friday dates
        '''
        for col in self.text_cols:
            data.loc[:, col] = data[col] + ' [SEP] '

        dfs = []
        for ticker in tqdm(self.relevant_tickers):
            aggregated = data[data['relatedTickers'].apply(
                lambda x: ticker in x)].resample('W-fri', on='providerPublishTime').sum()
            # aggregated = aggregated.drop('relatedTickers', axis=1)
            aggregated['ticker'] = ticker
            aggregated = aggregated.drop_duplicates('ticker', keep='last')
            if aggregated.empty:
                continue
            dfs.append(aggregated)
        new_df = pd.concat(dfs)

        new_df['title'] = new_df['title'].astype(str)
        # merged = new_df.merge(self.tickers, on='ticker')
        # merged = merged.drop('yahoo', axis=1).dropna()
        return new_df

    def full_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        '''
        Reads in API data and makes it ready for further analysis

        :param data: Pandas DataFrame generated with stocknewsapi_loader
        :return: Data grouped by ticker and friday dates
        '''
        proc_data = self.preprocess(data)
        agg_data = self.aggregate(proc_data)
        agg_data['relatedTickers'] = agg_data['relatedTickers'].apply(set)
        return agg_data.merge(self.pre_data)



# Model inference parameters
MAX_LEN = 256
BATCH_SIZE = 8

# Set seed for reproducability
seed = 5321
rn.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# Surpress Pandas warnings
pd.set_option('chained_assignment', None)


# %%
finances = [Ticker(ticker) for ticker in financial_data]
news = []
infos = []
for finance in tqdm(finances):
    finanace:Ticker
    news_data:dict = finance.news
    preload_detail = finance.fast_info
    detail_data = {
        "ticker": finance.ticker,
        "open": preload_detail.open,
        "close": preload_detail.previous_close,
    }
    news.extend(news_data)
    infos.append(detail_data)
print(f"{len(news)} of news loaded successfully")


# %%
df = pd.DataFrame(news)
df = df.dropna()
df.to_csv(NEWS_DATA_PATH)
print(f"{len(df)} of news saved")
# %%
fdf = pd.DataFrame(infos)
fdf.to_csv(FINANCIAL_DATA_PATH)

# %%
# Process and aggregate collected data
raw_data = pd.read_csv(NEWS_DATA_PATH, index_col=0)
financial_data = pd.read_csv(FINANCIAL_DATA_PATH, index_col=0)
snp = StockNewsProcessor(financial_data)
proc_data = snp.full_preprocessing(raw_data)

# %%
sample = proc_data.iloc[0]
print(f"Example of aggregated headlines for '{sample['ticker']}' stock:\n")
print(sample['title'])


# %%
# Predict signals for all preprocessed data
fbc = FinBertCased(max_len=MAX_LEN, batch_size=BATCH_SIZE,
                   model_path=MODEL_PATH, device=Device())
print(f"{fbc.device} device is selected")
proc_data.loc[:, 'signal'], proc_data['amount'] = fbc.predict_signals(proc_data['title'])


# %%
proc_data['trust_rate'] = proc_data['amount']/proc_data['amount'].sum() * 100
proc_data['score'] = proc_data['trust_rate'] * proc_data['signal']

# %%
print("Some of the predictions made by FinBERT:")
proc_data.sort_values(
    by='score', ascending=False)
proc_data.to_csv("results.csv")
proc_data.head()

# %%
buy = proc_data[proc_data['signal'] == proc_data['signal'].max()]
print(f"Stock we should buy this week: '{buy['ticker'].item()}'")
print(f"Signal: {buy['signal'].item()}")
print(f"\nNews headlines:")
for i, item in enumerate(buy['title'].item().split(' [SEP] ')[:-1]):
    print(f"{i+1}. {item}")


# %%
sell = proc_data[proc_data['signal'] == proc_data['signal'].min()]
print(f"Stock we should sell this week: '{sell['ticker'].item()}'")
print(f"Signal: {sell['signal'].item()}")
print(f"\nNews headlines:")
for i, item in enumerate(sell['title'].item().split(' [SEP] ')[:-1]):
    print(f"{i+1}. {item}")


# %%
plt.figure(figsize=(10, 5))
plt.title("Signal prediction distribution", weight='bold', fontsize=18)
plt.xlabel("score", fontsize=16)
plt.ylabel("Frequency", fontsize=16)
proc_data['score'].plot(kind='hist', bins=20)


# %%
report_data = proc_data.sort_values(by='score', ascending=False)
report_data['score'] = report_data['score'].round(2)
report_data['trust_rate'] = report_data['trust_rate'].round(2)
report_data['title'] = report_data['title'].apply(
    lambda x: ', '.join(x.split(' [SEP] ')[:2]))
report_data['open'] = report_data['open'].round(2)
report_data['close'] = report_data['close'].round(2)
report_data_positive = report_data.iloc[:5]
report_data_negative = report_data.iloc[-5:]

class Chat:
    def __init__(self, model="gpt-3.5-turbo", user="user", ai="assistant", default_prompt=""):
        self.model = model
        self.ai = ai
        self.user = user
        self.prompt = [{'role': 'system', 'content': default_prompt}]
    
    def length_of_text(self):
        return len(str(self.prompt))
    
    def insert_text(self, text):
        self.prompt.append({'role': self.user, 'content': text})
    
    def get_request(self):
        result = openai.ChatCompletion.create(
            model=self.model, messages=self.prompt)
        result_text = result['choices'][0]['message']['content']
        self.prompt.append({'role': self.ai, 'content': result_text})
        return result_text

# %%
prompt = "you are a blogger that writes the financial reports in korean. You can write reports from the financial dataset. the data will be provided below: titles of news, tickers, score and trust rates. you can't write ungiven data."

chat = Chat(ai="assistant", default_prompt=prompt)
chat.insert_text(f"오늘 날짜: {datetime.now()}")
chat.insert_text(f"긍정적인 주식 데이터: ```csv\n{report_data_positive[['ticker', 'title', 'score', 'trust_rate', 'open', 'close']].to_csv()}\n```")
chat.insert_text(f"부정적인 주식 데이터: ```csv\n{report_data_negative[['ticker', 'title', 'score', 'trust_rate', 'open', 'close']].to_csv()}\n```")
chat.insert_text(f"상기 데이터들을 이용해 markdown 문법에 맞추어 긍정, 부정적인 데이터에 대한 블로그 포스팅을 작성하라. 제목은 데이터에 기반해 보여주도록 한다. 포스트의 서론부는 주식에 대한 흥미를 일으키도록 한다. trust rate는 해당 데이터가 얼마나 신빙성이 있는지에 대한 데이터임을 주의하라. title은 뉴스의 제목으로, 포스트 작성 시 해석하여 score와 같이 표현한다. 주식마다 BlockQuote를 추가하여 묶어 표현한다. 보고서에 상세한 수치는 꼭 기재하도록 한다.")

# %%
length = chat.length_of_text()
print(f"텍스트 길이: {length}(추정 시간: {length/20.5:.1f}초)")
text = chat.get_request()

print(text)

# %%
with open(f'./reports/report_{datetime.now().strftime("%y%m%d")}.md', 'w', encoding='utf-8') as w:
    w.write(text)


