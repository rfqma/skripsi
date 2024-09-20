import streamlit as st
import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def load_model(file_path):
  with open(file_path, 'rb') as file:
    return pickle.load(file)

vectorizer = load_model('./models/ind_vectorizer_model.pkl')
inset_knn1 = load_model('./models/ind_inset_knn1_model.pkl')
inset_knn3 = load_model('./models/ind_inset_knn3_model.pkl')
inset_knn5 = load_model('./models/ind_inset_knn5_model.pkl')
inset_knn7 = load_model('./models/ind_inset_knn7_model.pkl')
inset_knn9 = load_model('./models/ind_inset_knn9_model.pkl')
sentistrength_knn1 = load_model('./models/ind_sentistrength_knn1_model.pkl')
sentistrength_knn3 = load_model('./models/ind_sentistrength_knn3_model.pkl')
sentistrength_knn5 = load_model('./models/ind_sentistrength_knn5_model.pkl')
sentistrength_knn7 = load_model('./models/ind_sentistrength_knn7_model.pkl')
sentistrength_knn9 = load_model('./models/ind_sentistrength_knn9_model.pkl')

SLANG_DICTIONARY_FILE_NAME_1 = "kamus_slang_1.csv"
SLANG_DICTIONARY_FILE_PATH_1 = f"dictionaries/{SLANG_DICTIONARY_FILE_NAME_1}"
DATA_FRAME_SLANG_DICTIONARY_1 = pd.read_csv(SLANG_DICTIONARY_FILE_PATH_1)

SLANG_DICTIONARY_FILE_NAME_2 = "kamus_slang_2.csv"
SLANG_DICTIONARY_FILE_PATH_2 = f"dictionaries/{SLANG_DICTIONARY_FILE_NAME_2}"
DATA_FRAME_SLANG_DICTIONARY_2 = pd.read_csv(SLANG_DICTIONARY_FILE_PATH_2)

SLANG_DICTIONARY_1 = pd.Series(DATA_FRAME_SLANG_DICTIONARY_1.formal.values, index=DATA_FRAME_SLANG_DICTIONARY_1.slang).to_dict()
SLANG_DICTIONARY_2 = pd.Series(DATA_FRAME_SLANG_DICTIONARY_2.formal.values, index=DATA_FRAME_SLANG_DICTIONARY_2.slang).to_dict()

SW_DICTIONARY_FILE_NAME_1 = "kamus_stopwords_1.csv"
SW_DICTIONARY_FILE_PATH_1 = f"dictionaries/{SW_DICTIONARY_FILE_NAME_1}"
DATA_FRAME_SW_DICTIONARY_1 = pd.read_csv(SW_DICTIONARY_FILE_PATH_1)

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

def drop_stopwords(text):
  custom_stopwords = stopwords.words('indonesian')
  custom_stopwords.clear()
  custom_stopwords.extend(DATA_FRAME_SW_DICTIONARY_1["stopwords"].values)

  factory = StopWordRemoverFactory()
  sastrawi_stopwords = factory.get_stop_words()

  combined_stopwords = set(custom_stopwords).union(set(sastrawi_stopwords))

  return " ".join([word for word in text.split() if word not in combined_stopwords])

def stem_indonesian_text(text):
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  return " ".join([stemmer.stem(word) for word in text.split()])

############################################################################
   
st.title('Analisis Sentimen dengan KNN berbasis Lexicon pada Bahasa Indonesia')
st.write("Analisis Reaksi Publik Terhadap Pemindahan Ibu Kota Negara Menggunakan Metode Lexicon Based dan KNN")
st.text('Masukkan kalimat terkait topik relokasi ibu kota dan dapatkan prediksi sentimennya')


model_selection = st.selectbox('Pilih model kamus lexicon dan nilai K', ['k1-InSet', 'k3-InSet', 'k5-InSet', 'k7-InSet', 'k9-InSet', 'k1-sentistrength_id', 'k3-sentistrength_id', 'k5-sentistrength_id', 'k7-sentistrength_id', 'k9-sentistrength_id'])

if model_selection == 'k1-InSet':
  sentiment_model = inset_knn1
elif model_selection == 'k3-InSet':
  sentiment_model = inset_knn3
elif model_selection == 'k5-InSet':
  sentiment_model = inset_knn5
elif model_selection == 'k7-InSet':
  sentiment_model = inset_knn7
elif model_selection == 'k9-InSet':
  sentiment_model = inset_knn9
elif model_selection == 'k1-sentistrength_id':
  sentiment_model = sentistrength_knn1
elif model_selection == 'k3-sentistrength_id':
    sentiment_model = sentistrength_knn3
elif model_selection == 'k5-sentistrength_id':
   sentiment_model = sentistrength_knn5
elif model_selection == 'k7-sentistrength_id':
   sentiment_model = sentistrength_knn7
elif model_selection == 'k9-sentistrength_id':
    sentiment_model = sentistrength_knn9


user_input = st.text_area('Masukkan kalimat di sini:')

if st.button('Prediksi Sentimen'):
  if user_input:
    # clean, standardize, translate, remove stopwords, stem text
    cleaned_text = clean_text(user_input)
    standardized_text = slang_dict_integration(cleaned_text)
    stopwords_removed_text = drop_stopwords(standardized_text)
    after_stemming_text = stem_indonesian_text(stopwords_removed_text)
    st.write(f'teks yang sudah diproses: **{after_stemming_text}**')

    # vectorize text
    vectorized_text = vectorizer.transform([after_stemming_text])
    st.write(f'teks yang sudah divektorisasi: **{vectorized_text}**')

    # predict sentiment
    sentiment = sentiment_model.predict(vectorized_text)[0]

    st.write(f'Sentimen dari kalimat yang dimasukkan: **{sentiment}**')
  else:
    st.write('Anda belum memasukkan kalimat.')