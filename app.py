import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

st.title("Fake News Detection App")

filepath="/content/fake_or_real_news.csv"
data=pd.read_csv(filepath, sep=',', delimiter=None, header='infer', names=None, 
            index_col=None, usecols=None, engine=None)

data['label']=np.where(data['label']=="FAKE",1,0)

data.rename(columns={'label':'fake'}, inplace=True)

X,y=data['text'],data['fake']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

vectorizer=TfidfVectorizer(stop_words="english",max_df=0.7)
X_train_vectorized=vectorizer.fit_transform(X_train)
X_test_vectorized=vectorizer.transform(X_test)

clf=LinearSVC()

clf.fit(X_train_vectorized, y_train)

print(clf.score(X_test_vectorized,y_test))

mytext=st.text_input("Enter the news you want to verify: ")
vectorized_text=vectorizer.transform([mytext])

# print(type(clf.predict(vectorized_text)))
def check_news(text):  
  if clf.predict(vectorized_text)==np.array(1):
    st.write("This is Fake News.. Beware!!")
  else:
    st.write("This news is accurate!")

if mytext:

  check_news(mytext)






