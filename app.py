import streamlit as st
import os
import sys
import asyncio
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import nest_asyncio
from dotenv import load_dotenv

from alpha_vantage.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import Dense, LSTM

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsApiClient

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
nest_asyncio.apply()

# Load .env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Google LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, max_output_tokens=1024)
def get_google_embeddings():
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# Top App UI

st.title("📊 Stock Sage")

tab = st.radio("Choose Feature", ["📈 Stock Profit Prediction", "📰 URLyBot News Research"])

# Conditionally show sidebar only for URLyBot
urls, process_url_clicked = [], False
if tab == "📰 URLyBot News Research":
    with st.sidebar:
        st.title("News Article URLs")
        for i in range(3):
            url = st.text_input(f"URL {i+1}", key=f"url_{i}")
            urls.append(url)
        process_url_clicked = st.button("Process URLs", key="process_urls")


# Tab 1: Stock Forecast + Profit


NEWS_API = os.getenv("NEWS_API")
ALPHA_API = os.getenv("ALPHA_API")

if tab == "📈 Stock Profit Prediction":
    st.header("📈 Stock Price Forecast + Profit Probability")
    ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL, MSFT)", "AAPL")
    buy_clicked = st.button("Buy", key="buy")
    sell_clicked = st.button("Sell", key="sell")

    @st.cache_data(show_spinner=False)
    def get_stock_data(ticker):
        api_key = ALPHA_API
        ts = TimeSeries(key=api_key, output_format='pandas')
        df, _ = ts.get_daily(symbol=ticker, outputsize='full')
        df = df.rename(columns={
            '1. open': 'Open', '2. high': 'High', '3. low': 'Low',
            '4. close': 'Close', '5. volume': 'Volume'
        })
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.loc['2018-01-01':(datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')]
        return df

    df = get_stock_data(ticker)
    if df is None or df.empty:
        st.error("Error fetching data. Please check the ticker.")
        st.stop()

    def train_model(df):
        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):])
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(data_training)

        x_train, y_train = [], []
        for i in range(100, len(scaled_train)):
            x_train.append(scaled_train[i-100:i])
            y_train.append(scaled_train[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(60, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(60, return_sequences=True))
        model.add(LSTM(60))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=20, batch_size=64, verbose=0)

        past_100_days = data_training.tail(100)
        final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
        input_data = scaler.transform(final_df)

        x_test, y_test = [], []
        for i in range(100, input_data.shape[0]):
            x_test.append(input_data[i-100:i])
            y_test.append(input_data[i, 0])
        x_test, y_test = np.array(x_test), np.array(y_test)

        y_predicted = model.predict(x_test)
        y_predicted = scaler.inverse_transform(y_predicted)
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        return model, y_test, y_predicted, scaler

    model, y_test, y_predicted, scaler = train_model(df)

    def get_sentiment(company_name):
        news_api = NewsApiClient(api_key=NEWS_API)
        sia = SentimentIntensityAnalyzer()
        from_date = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
        to_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

        articles = news_api.get_everything(q=company_name,
                                           from_param=from_date,
                                           to=to_date,
                                           language='en',
                                           sort_by='relevancy',
                                           page_size=50)

        compound_scores = []
        for article in articles['articles']:
            title = article.get('title', '') or ''
            description = article.get('description', '') or ''
            text = title + ". " + description
            score = sia.polarity_scores(text)['compound']
            compound_scores.append(score)

        return np.mean(compound_scores) if compound_scores else 0.0

    avg_sentiment = get_sentiment(ticker)

    current_price = df['Close'].iloc[-1]
    predicted_price = y_predicted[-1][0]
    normalized_change = (predicted_price - current_price) / current_price

    base_prob = 50 + (normalized_change * 100)
    base_prob = max(5, min(95, base_prob))
    sentiment_adjustment = avg_sentiment * 25
    profit_probability = base_prob + sentiment_adjustment if normalized_change > 0 else base_prob - sentiment_adjustment
    profit_probability = max(5, min(95, profit_probability))

    if buy_clicked:
        st.subheader("📈 Buy Profit Probability")
        st.success("Based on price trend and sentiment")
        st.slider("Chances of Profit if You Buy", 0, 100, int(profit_probability), disabled=True)

    if sell_clicked:
        st.subheader("📉 Sell Profit Probability")
        st.error("Based on price trend and sentiment")
        st.slider("Chances of Profit if You Sell", 0, 100, int(100 - profit_probability), disabled=True)

    with st.expander("📊 Model Evaluation Metrics"):
        st.write(f"**Current Price**: ${current_price:.2f}")
        st.write(f"**Predicted Price**: ${predicted_price:.2f}")
        st.write(f"**Avg News Sentiment**: {avg_sentiment:.3f}")
        rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
        mae = mean_absolute_error(y_test, y_predicted)
        r2 = r2_score(y_test, y_predicted)
        actual_direction = np.sign(y_test[1:] - y_test[:-1])
        predicted_direction = np.sign(y_predicted[1:] - y_predicted[:-1])
        directional_accuracy = np.mean(actual_direction == predicted_direction)
        # st.write(f"**Directional Accuracy**: {directional_accuracy * 100:.2f}%")


# Tab 2: URLyBot

elif tab == "📰 URLyBot News Research":
    st.header("📰 URLyBot: News Research Tool")
    vectorstore_path = "faiss_gemini_store"
    main_placeholder = st.empty()

    if process_url_clicked:
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Loading data from URLs...")
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)

        embeddings = get_google_embeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        main_placeholder.text("Building vector embeddings...")
        time.sleep(2)
        vectorstore.save_local(vectorstore_path)

    query = main_placeholder.text_input("Ask a question about the articles:")
    if query:
        if os.path.exists(vectorstore_path):
            embeddings = get_google_embeddings()
            vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)

            retriever = vectorstore.as_retriever()
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            result = chain({"question": query}, return_only_outputs=True)

            st.subheader("📌 Answer")
            st.write(result["answer"])
            if result.get("sources", ""):
                st.subheader("🔗 Sources")
                for source in result["sources"].split("\n"):
                    st.write(source)
