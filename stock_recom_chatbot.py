import streamlit as st
import requests
import torch
import pandas as pd
import mplfinance as mpf
from loguru import logger
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import FinanceDataReader as fdr
from langchain_openai import OpenAIEmbeddings

def get_news(company_name, client_id, client_secret):
    """네이버 뉴스 API를 사용하여 최신 뉴스 가져오기"""
    url = f"https://openapi.naver.com/v1/search/news.json?query={company_name}&display=5&sort=date"
    headers = {"X-Naver-Client-Id": client_id, "X-Naver-Client-Secret": client_secret}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()["items"]
    else:
        return []


def process_news_data(news_items):
    """RecursiveCharacterTextSplitter를 활용하여 뉴스 조각내기"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = [news['title'] + " " + news['description'] for news in news_items]
    split_texts = text_splitter.split_text(" ".join(texts))
    return split_texts


def create_embeddings(texts):
    """openai embeddings 활용하여 뉴스 임베딩 생성 및 저장"""
    model_name = "jhgan/ko-sbert-nli"
    encode_kwargs = {'normalize_embeddings': True}

    ko_embedding = HuggingFaceEmbeddings(
        model_name_or_path=model_name,  # 최신 버전에서는 model_name 대신 model_name_or_path 사용
        encode_kwargs=encode_kwargs
    )
    # VectorDB에 저장
    vectorstore = Chroma.from_documents(documents=splits, embedding=ko_embedding)
    return vector_store


def retrieve_relevant_sentences(query, vector_store):
    """LangChain Retriever를 사용하여 관련 문장 검색"""
    retriever = vector_store.as_retriever()
    relevant_docs = retriever.get_relevant_documents(query)
    return " ".join([doc.page_content for doc in relevant_docs])


def generate_summary(query, context, openai_key):
    """GPT-4를 활용하여 요약 생성"""
    llm = ChatOpenAI(openai_api_key=openai_key, model_name='gpt-4', temperature=0)
    qa_chain = RetrievalQA(llm=llm, retriever=context)
    return qa_chain.run(query)


def analyze_sentiment(news_text):
    """KoBERT 모델을 사용하여 뉴스 감성 분석"""
    tokenizer = AutoTokenizer.from_pretrained("skt/kobert-base-v1")
    model = AutoModelForSequenceClassification.from_pretrained("skt/kobert-base-v1")

    inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment = "긍정" if probs[0][1] > probs[0][0] else "부정"
    return sentiment


def visualize_stock(symbol, period):
    """MPLFinance를 활용한 주가 시각화 (일/주/월/년)"""
    df = fdr.DataReader(symbol, '2024-01-01')
    if period == "일":
        df = df.tail(30)
    elif period == "주":
        df = df.resample('W').last()
    elif period == "월":
        df = df.resample('M').last()
    elif period == "년":
        df = df.resample('Y').last()
    mpf.plot(df, type='candle', style='charles', title=f"{symbol} 주가 ({period})", volume=True)


st.title("국내 주식 뉴스 기반 추천 QA 챗봇")
company_name = st.text_input("기업명을 입력하세요:")
client_id = "gPLyAEqy9ENxbExk8LJP"
client_secret = "Ogu99du1xG"
openai_key = st.text_input("OpenAI API Key", type="password")
period = st.selectbox("조회 기간", ["일", "주", "월", "년"])

if st.button("검색"):
    if not openai_key:
        st.info("OpenAI API 키를 입력해주세요.")
        st.stop()
    news_items = get_news(company_name, client_id, client_secret)
    if news_items:
        st.subheader("최신 뉴스 및 감성 분석 결과")
        processed_texts = process_news_data(news_items)
        vector_store = create_embeddings(processed_texts)
        context = retrieve_relevant_sentences(company_name, vector_store)
        summary = generate_summary(company_name, context, openai_key)
        st.write("요약 정보:", summary)
        for news in news_items:
            sentiment = analyze_sentiment(news['title'] + " " + news['description'])
            st.write(f"[{news['title']}]({news['link']}) - 감성 분석: {sentiment}")
        st.subheader("주가 트렌드")
        symbol = fdr.StockListing("KRX")[fdr.StockListing("KRX")["Name"] == company_name]["Code"].values
        if len(symbol) > 0:
            visualize_stock(symbol[0], period)
        else:
            st.write("해당 기업의 주식 정보를 찾을 수 없습니다.")
    else:
        st.write("관련 뉴스가 없습니다.")
