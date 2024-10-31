import re
import pandas as pd
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from twikit import Client, TooManyRequests
import asyncio
from datetime import datetime
import time
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#################################################################################

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

NEGASI_DICTIONARY_FILE_NAME_1 = "negasi.csv"
NEGASI_DICTIONARY_FILE_PATH_1 = f"dictionaries/{NEGASI_DICTIONARY_FILE_NAME_1}"
DATA_FRAME_NEGASI_DICTIONARY_1 = pd.read_csv(NEGASI_DICTIONARY_FILE_PATH_1)

ANTONYM_DICTIONARY_FILE_NAME_1 = "antonim_bahasa_indonesia.csv"
ANTONYM_DICTIONARY_FILE_PATH_1 = f"dictionaries/{ANTONYM_DICTIONARY_FILE_NAME_1}"
DATA_FRAME_ANTONYM_DICTIONARY_1 = pd.read_csv(ANTONYM_DICTIONARY_FILE_PATH_1)
ANTONYM_DICTIONARY_1 = pd.Series(DATA_FRAME_ANTONYM_DICTIONARY_1.antonim.values, index=DATA_FRAME_ANTONYM_DICTIONARY_1.word).to_dict()

#################################################################################

with open("./lexicons/lexicon_json_inset-neg.txt") as f:
  inset_neg = f.read()
with open("./lexicons/lexicon_json_inset-pos.txt") as f:
  inset_pos = f.read()

insetNeg = json.loads(inset_neg)
insetPos = json.loads(inset_pos)

sia_inset_neg = SentimentIntensityAnalyzer()
sia_inset_pos = SentimentIntensityAnalyzer()

sia_inset_neg.lexicon.clear()
sia_inset_pos.lexicon.clear()

sia_inset_neg.lexicon.update(insetNeg)
sia_inset_pos.lexicon.update(insetPos)

#################################################################################

def load_model(file_path):
  with open(file_path, 'rb') as file:
    return joblib.load(file)

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

def slang_dict_integration_kamus_1(text):
  words = text.split()
  standardization_words = []

  for word in words:
    if word in SLANG_DICTIONARY_1:
      standardization_words.append(SLANG_DICTIONARY_1[word])
    else:
      standardization_words.append(word)
  return " ".join(standardization_words)

def slang_dict_integration_kamus_2(text):
  words = text.split()
  standardization_words = []

  for word in words:
    if word in SLANG_DICTIONARY_2:
      standardization_words.append(SLANG_DICTIONARY_2[word])
    else:
      standardization_words.append(word)
  return " ".join(standardization_words)

def underscore_negation(text):
  words = text.split()
  negation_words = set(DATA_FRAME_NEGASI_DICTIONARY_1["negasi"].values)
  skip_next = False
  new_words = []
    
  for i in  range(len(words)):
    if skip_next:
      skip_next = False
      continue
    if words[i] in negation_words and i < len(words) - 1:
      new_words.append(words[i] + "_" + words[i+1])
      skip_next = True
    else:
      new_words.append(words[i])
  return " ".join(new_words)

def swap_antonyms(text):
  words = text.split()
  antonym_dict = dict(zip(DATA_FRAME_ANTONYM_DICTIONARY_1["word"], DATA_FRAME_ANTONYM_DICTIONARY_1["antonim"]))
  new_words = []
    
  for word in words:
    if "_" in word:
      negation, next_word = word.split("_", 1)
      if next_word in antonym_dict:
        new_words.append(antonym_dict[next_word])
      else:
        new_words.append(word)
    else:
      new_words.append(word)
  return " ".join(new_words)

def replace_underscore(text):
  text = re.sub(r'_', ' ', text)
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

def get_model_evaluation(size_test, ratio, k):
  ds = pd.read_csv("outputs/sentiment.csv")

  tfidf = TfidfVectorizer()
  X_tfidf = tfidf.fit_transform(ds["preprocessed_text"])

  smote = SMOTE(random_state=21)
  X_smote, Y_smote = smote.fit_resample(X_tfidf, ds["sentiment_label"])

  X_train,X_test,Y_train,Y_test=train_test_split(X_smote, Y_smote, test_size=size_test, random_state=21)

  knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
  knn.fit(X_train,Y_train)
  y_pred = knn.predict(X_test)

  accuracy = accuracy_score(Y_test, y_pred)
  precision = precision_score(Y_test, y_pred, average='weighted')
  recall = recall_score(Y_test, y_pred, average='weighted')
  return {"accuracy": accuracy, "precision": precision, "recall": recall, "Y_test": Y_test, "y_pred": y_pred, "labels": knn.classes_}

async def get_tweets(query, client):
  print(f'{datetime.now()} - Fetching tweets...')
  try:
    tweets = await client.search_tweet(query, product="Latest", count=5)
  except TooManyRequests as e:
    rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
    print(f'{datetime.now()} - Rate limit reached. Waiting until {rate_limit_reset}...')
    wait_time = rate_limit_reset - datetime.now()
    time.sleep(wait_time.total_seconds())
    return []
  
  tweet_data = []
  for idx, tweet in enumerate(tweets, 1):
      tweet_info = {
          '#': idx,
          'tweet_id': tweet.id,
          'user_id': tweet.user.id,
          'username': tweet.user.screen_name,
          'display_name': tweet.user.name,
          'text': tweet.text,
          'created_at': tweet.created_at,
          'retweet_count': tweet.retweet_count,
          'favorite_count': tweet.favorite_count
      }

      tweet_data.append(tweet_info)
      if idx >= 5:
          break

  print(f'{datetime.now()} - {len(tweet_data)} tweets fetched successfully.')
  return tweet_data

def call_get_tweets(query):
  user_agent = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Mobile Safari/537.36'
  client = Client(language='en-US', user_agent=user_agent)
  client.load_cookies('scraper/twikit_cookies.json')
  tweets = asyncio.run(get_tweets(query, client))
  return pd.DataFrame(tweets)

def get_sentiment(text):
  neg = sia_inset_neg.polarity_scores(text)["compound"]
  pos = sia_inset_pos.polarity_scores(text)["compound"]
  inset_compound_score = neg + pos
  if inset_compound_score > 0:
    return {"sentiment": "positif", "compound_score": inset_compound_score, "positive_score": pos, "negative_score": neg}
  elif inset_compound_score < 0:
    return {"sentiment": "negatif", "compound_score": inset_compound_score, "positive_score": pos, "negative_score": neg}
  else:
    return {"sentiment": "netral", "compound_score": inset_compound_score, "positive_score": pos, "negative_score": neg}