import pickle as pkl
import streamlit as st
import preprocess_kgptalkie as ps
import re

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000)


def get_clean(x):
    x = str(x).lower().replace('\\', '').replace('_', ' ')
    x = ps.cont_exp(x)
    x = ps.remove_emails(x)
    x = ps.remove_urls(x)
    # x = ps.remove_html_tags(x)
    x = ps.remove_accented_chars(x)
    x = ps.remove_special_chars(x)
    x = re.sub("(.)\\1{2,}", "\\1", x)
    return x


# Loading the  pickle model we have already trained

clf = pkl.load(open('model.pkl', 'rb'))


st.title("Movie sentiment Analysis")
st.subheader('ML Project')
st.write('This project is based on NLP and Support vector machines')

message = st.text_area("Enter Text", "Type Here ..")

if st.button("Predict"):
    disp = ""
    message = get_clean(message)
    op = clf.predict([message])
    if(op[0] == 1):
        disp = "Positive Review"
    else:
        disp = "Negative Review"
    st.title(disp)
