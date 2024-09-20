import streamlit as st
import pandas as pd
import pickle
import re
from deep_translator import GoogleTranslator
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

with open('models/eng_vectorizer_model.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

with open('models/eng_knn1_model.pkl', 'rb') as model_file:
    knn1 = pickle.load(model_file)

with open('models/eng_knn3_model.pkl', 'rb') as model_file:
    knn3 = pickle.load(model_file)

with open('models/eng_knn5_model.pkl', 'rb') as model_file:
    knn5 = pickle.load(model_file)

with open('models/eng_knn7_model.pkl', 'rb') as model_file:
    knn7 = pickle.load(model_file)

SLANG_DICTIONARY_FILE_NAME_1 = "kamus_slang_1.csv"
SLANG_DICTIONARY_FILE_PATH_1 = f"dictionaries/{SLANG_DICTIONARY_FILE_NAME_1}"
DATA_FRAME_SLANG_DICTIONARY_1 = pd.read_csv(SLANG_DICTIONARY_FILE_PATH_1)

SLANG_DICTIONARY_FILE_NAME_2 = "kamus_slang_2.csv"
SLANG_DICTIONARY_FILE_PATH_2 = f"dictionaries/{SLANG_DICTIONARY_FILE_NAME_2}"
DATA_FRAME_SLANG_DICTIONARY_2 = pd.read_csv(SLANG_DICTIONARY_FILE_PATH_2)

SLANG_DICTIONARY_1 = pd.Series(DATA_FRAME_SLANG_DICTIONARY_1.formal.values, index=DATA_FRAME_SLANG_DICTIONARY_1.slang).to_dict()
SLANG_DICTIONARY_2 = pd.Series(DATA_FRAME_SLANG_DICTIONARY_2.formal.values, index=DATA_FRAME_SLANG_DICTIONARY_2.slang).to_dict()

def slang_dict_integration(text):
  words = text.split()
  standardization_words = []

  for word in words:
    if word in SLANG_DICTIONARY_1:
      standardization_words.append(SLANG_DICTIONARY_1[word])
    elif word in SLANG_DICTIONARY_2:
      standardization_words.append(SLANG_DICTIONARY_2[word])
    else:
      standardization_words.append(word)

  return " ".join(standardization_words)

def clean_text(text):
  # remove RT tag
  text = re.sub(r'RT\s', '', text)
  # remove @_username
  text = re.sub(r"\@([\w]+)", " ", text)
  # replace emoji decode with space
  text = re.sub(r"\\u[a-zA-Z0-9]{4}", " ", text)
  # replace enter /n/ with space
  text = re.sub(r"\n\s", " ", text)
  text = re.sub(r"\n", " ", text)
  # remove non-ascii
  text = re.sub(r'[^\x00-\x7F]+',' ', text)
  # fix duplicate characters (ex: hellooooo)
  text = re.sub(r'([a-zA-Z])\1\1','\\1', text)
  # replace url
  text = re.sub(r'http[s]?\:\/\/.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+',' ', text)
  text = re.sub(r'pic.twitter.com?.[a-zA-Z0-9\.\/\_?=%&#\-\+!]+',' ', text)
  # convert to lowercase
  text = text.lower()
  # remove hashtag
  text = re.sub(r'\#[a-zA-Z0-9_]+','', text)
  # remove numbers
  text = re.sub(r'[0-9]+',' ', text)
  # remove symbols
  text = re.sub(r'[!$%^&*@#()_+|~=`{}\[\]%\-:";\'<>?,.\/]', ' ', text)
  # remove extra spaces to one space
  text = re.sub(r' +', ' ', text)
  # remove leading and trailing spaces
  text = re.sub(r'^[ ]|[ ]$','', text)
  # replace ikn with ibu kota negara baru
  text = text.replace("ikn", "ibu kota negara baru")
  
  return text

def translate_text(text):
  return GoogleTranslator(source='id', target='en').translate(text)

def drop_stopwords(text):
  stop_words = set(stopwords.words('english'))
  return " ".join([word for word in text.split() if word not in stop_words])

def lemmatize_text(text):
  lemmatizer = WordNetLemmatizer()
  return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

##############################################################################
   
st.title('Sentimen Analisis dengan KNN & Lexicon Based')
st.write('Masukkan kalimat terkait topik relokasi ibu kota dan dapatkan prediksi sentimennya.')

model_selection = st.selectbox('Select K value', ['k1', 'k3', 'k5', 'k7'])

if model_selection == 'k1':
  sentiment_model = knn1
elif model_selection == 'k3':
  sentiment_model = knn3
elif model_selection == 'k5':
  sentiment_model = knn5
elif model_selection == 'k7':
  sentiment_model = knn7

user_input = st.text_area('Enter your sentence here:')

if st.button('Predict Sentiment'):
  if user_input:
    # clean, standardize, translate, remove stopwords, lemmatize text
    cleaned_text = clean_text(user_input)
    standardized_text = slang_dict_integration(cleaned_text)
    translated_text = translate_text(standardized_text)
    final_cleaned_text = clean_text(translated_text)
    stopwords_removed_text = drop_stopwords(final_cleaned_text)
    lemmatized_text = lemmatize_text(stopwords_removed_text)
    st.write(f'preprocessed text: **{lemmatized_text}**')

    # vectorize text
    vectorized_text = vectorizer.transform([lemmatized_text])
    st.write(f'vectorized text: **{vectorized_text}**')

    # predict sentiment
    sentiment = sentiment_model.predict(vectorized_text)[0]

    st.write(f'Sentiment of the entered sentence: **{sentiment}**')
  else:
    st.write('Please enter a sentence.')