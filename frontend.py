# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:59:38 2021

@author: VRUTIKA
"""

import warnings
warnings.filterwarnings('ignore')
# imports
import pandas as pd
# matplotlib 
import matplotlib.pyplot as plt
# sns
import seaborn as sns
# plotly ex
import plotly.express as px

# import streamlit
import streamlit as st
from wordcloud import WordCloud, STOPWORDS

from newspaper import Article

import pickle
filename = 'C:/Python/ML2/researchproject/data/model.pkl'
model = pickle.load(open(filename, 'rb'))

filename = 'C:/Python/ML2/researchproject/data/vars.pkl'
fe = pickle.load(open(filename, 'rb'))

def get_article(url):
    toi_article = Article(url, language="en")
    toi_article.download()
    toi_article.parse()
    toi_article.nlp()
    article = toi_article.text
    
    return article

def get_category(article):
    
    tf = fe.transform([article])
    category = model.predict(tf)
    
    return category[0]
    
def wc(article):
    wordcloud = WordCloud(stopwords = STOPWORDS, background_color='white', height=600, width=600).generate(article)
    plt.imshow(wordcloud,interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    col2.pyplot()

# title
st.title("News Article Classification Model")

# Navigation

status = st.sidebar.radio("Navigation", ('Predict', 'Documentation'))
if (status == 'Documentation'):
    
    fn = ""
    fn = st.selectbox("Files:",
                    ['model.py', 'frontend.py','data'])
 
    if fn == 'data':
        file = "C:/Python/ML2/researchproject/data/bbc-text.csv"
        df = pd.read_csv(file)
        st.dataframe(df)
    else:
        file = "C:/Python/ML2/researchproject/"+fn
        f = open(file, "r")
        code = f.read()
        st.code(code, language='python')
else:
    
    col1,col2=st.columns([1,1])
    col1.subheader("Prediction:")
    col2.subheader("WordCloud:")
    
    status = st.sidebar.radio("INPUT:", ('URL', 'Article'))
    
    if (status == 'URL'):
        url = st.sidebar.text_input("Enter URL", "")
        if(st.sidebar.button('Submit')):
            if url == "":
                st.markdown("Please enter URL")
            else:
                try:
                    article = get_article(url)
                    category = get_category(article)
                    col1.text("Predicted Category:{}".format(category))
                    col1.info(article)
                    wc(article)
                except:
                    st.markdown("Please make sure URL is correct.")                  
                
            
    else:
        article = st.sidebar.text_area("Enter Article","")
        if(st.sidebar.button('Submit')):
            if article == "":
                st.markdown("Please enter article")
            else:
                category = get_category(article)
                col1.text("Predicted Category:{}".format(category))
                col1.info(article)
                wc(article)
