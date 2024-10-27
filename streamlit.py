import streamlit as st
import stfunc
import pandas as pd

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("streamlit.css")

knn_model = stfunc.load_model('models/8020/knn1_8020_model.pkl')
tfidf_vectorizer = stfunc.load_model('models/tf-idf/tfidf_vectorizer.pkl')

if "page" not in st.session_state:
  st.session_state.page = "Beranda"

st.sidebar.title('Menu')
if st.sidebar.button('Beranda'):
  st.session_state.page = "Beranda"
if st.sidebar.button('Prediksi'):
  st.session_state.page = "Prediksi"
if st.sidebar.button('Dataset'):
  st.session_state.page = "Dataset"
if st.sidebar.button('Evaluasi'):
  st.session_state.page = "Evaluasi"

if st.session_state.page == "Beranda":
  st.title("Penerapan Metode Klasifikasi KNN dan Lexicon Based dengan Ekstraksi Fitur TF-IDF untuk Analisis Sentimen Publik Terhadap Pemindahan Ibu Kota Negara")

if st.session_state.page == "Prediksi":
  st.text('Masukkan kalimat terkait topik relokasi ibu kota dan dapatkan prediksi sentimennya')
  user_input = st.text_area('Masukkan kalimat di sini:')
  if st.button('Prediksi Sentimen'):
    if user_input:
      # clean, standardize, negation handling, remove stopwords, stem text
      cleaned_text = stfunc.clean_text(user_input)
      standardized_text = stfunc.slang_dict_integration_kamus_1(cleaned_text)
      standardized_text = stfunc.slang_dict_integration_kamus_2(standardized_text)
      underscore_negation_text = stfunc.underscore_negation(standardized_text)
      swap_negation_text = stfunc.swap_antonyms(underscore_negation_text)
      final_negation_text = stfunc.replace_underscore(swap_negation_text)
      stopwords_removed_text = stfunc.drop_stopwords(final_negation_text)
      after_stemming_text = stfunc.stem_indonesian_text(stopwords_removed_text)
      st.write(f'Kalimat setelah di preprocessing: **{after_stemming_text}**')

      # vectorize text
      vectorized_text = tfidf_vectorizer.transform([after_stemming_text])
      st.write(f'vectorized text: {vectorized_text}')

      # predict sentiment
      sentiment = knn_model.predict(vectorized_text)[0]
      st.write(f'Sentimen dari kalimat yang dimasukkan: **{sentiment}**')
    else:
      st.write('Anda belum memasukkan kalimat.')

if st.session_state.page == "Dataset":
  st.title('Dataset')
  st.text('Dataset yang digunakan dalam proses pelatihan model algoritma klasifikasi KNN')

  st.write('Fresh Dataset')
  df_fresh = pd.read_csv("datasets/merged/merged_dataset.csv")
  st.dataframe(df_fresh)

  st.write('Dataset after preprocessing')
  df_preprocessed = pd.read_csv("outputs/preprocessed.csv")
  st.dataframe(df_preprocessed)

  st.write('Dataset after sentiment labelling')
  df_labelled = pd.read_csv("outputs/sentiment.csv")
  st.dataframe(df_labelled)

if st.session_state.page == "Evaluasi":
  st.title('Evaluasi')