import streamlit as st
import stfunc
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

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
  st.title("Penerapan Metode Klasifikasi K-Nearest Neighbor dengan Ekstraksi Fitur TF-IDF untuk Analisis Sentimen Publik Berbasis Leksikon Terhadap Pemindahan Ibu Kota Negara")

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
  st.text('Evaluasi model algoritma klasifikasi KNN')
  eval = st.selectbox('Pilih parameter:', ['k=1', 'k=3', 'k=5'])

  if eval == 'k=1':
    __json = stfunc.get_model_evaluation(0.2, "80:20", 1)
    st.write(__json)
    acc = __json['accuracy'] * 100
    pre = __json['precision'] * 100
    rec = __json['recall'] * 100

    st.write('### Metriks')
    st.write('Accuracy:', f"{acc:.2f}%")
    st.write('Precision:', f"{pre:.2f}%")
    st.write('Recall:', f"{rec:.2f}%")
    metrics_df = pd.DataFrame({
      'Metric': ['Accuracy', 'Precision', 'Recall'],
      'Score': [__json['accuracy'], __json['precision'], __json['recall']]
    })

    st.bar_chart(metrics_df.set_index('Metric'))
    st.write('### Confusion Matrix')
    cm = confusion_matrix(__json['Y_test'], __json['y_pred'], labels=__json['labels'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=__json['labels'])
    disp.plot()
    plt.title(f"Confusion Matrix for k=1")
    st.pyplot(plt)
  elif eval == 'k=3':
    __json = stfunc.get_model_evaluation(0.2, "80:20", 3)
    st.write(__json)
    acc = __json['accuracy'] * 100
    pre = __json['precision'] * 100
    rec = __json['recall'] * 100

    st.write('### Metriks')
    st.write('Accuracy:', f"{acc:.2f}%")
    st.write('Precision:', f"{pre:.2f}%")
    st.write('Recall:', f"{rec:.2f}%")
    metrics_df = pd.DataFrame({
      'Metric': ['Accuracy', 'Precision', 'Recall'],
      'Score': [__json['accuracy'], __json['precision'], __json['recall']]
    })

    st.bar_chart(metrics_df.set_index('Metric'))
    st.write('### Confusion Matrix')
    cm = confusion_matrix(__json['Y_test'], __json['y_pred'], labels=__json['labels'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=__json['labels'])
    disp.plot()
    plt.title(f"Confusion Matrix for k=3")
    st.pyplot(plt)
  elif eval == 'k=5':
    __json = stfunc.get_model_evaluation(0.2, "80:20", 5)
    st.write(__json)
    acc = __json['accuracy'] * 100
    pre = __json['precision'] * 100
    rec = __json['recall'] * 100

    st.write('### Metriks')
    st.write('Accuracy:', f"{acc:.2f}%")
    st.write('Precision:', f"{pre:.2f}%")
    st.write('Recall:', f"{rec:.2f}%")
    metrics_df = pd.DataFrame({
      'Metric': ['Accuracy', 'Precision', 'Recall'],
      'Score': [__json['accuracy'], __json['precision'], __json['recall']]
    })

    st.bar_chart(metrics_df.set_index('Metric'))
    st.write('### Confusion Matrix')
    cm = confusion_matrix(__json['Y_test'], __json['y_pred'], labels=__json['labels'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=__json['labels'])
    disp.plot()
    plt.title(f"Confusion Matrix for k=5")
    st.pyplot(plt)
    