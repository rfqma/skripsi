from twikit import Client, TooManyRequests
import asyncio
from datetime import datetime
import csv
from random import randint
import time

MIN_TWEETS = 1000

QUERY_FIRST = 'ikn lang:id since:2024-01-01 until:2024-10-01'
QUERY_SECOND = 'ibu kota baru lang:id since:2024-01-01 until:2024-10-01'
QUERY_THIRD = 'ibu kota nusantara lang:id since:2024-01-01 until:2024-10-01'
QUERY_FOURTH = 'ibu kota pindah lang:id since:2024-01-01 until:2024-10-01'
QUERY_FIFTH = 'pemindahan ibu kota lang:id since:2024-01-01 until:2024-10-01'
QUERY_SIXTH = 'ibukota baru lang:id since:2024-01-01 until:2024-10-01'
QUERY_SEVENTH = 'ibukota nusantara lang:id since:2024-01-01 until:2024-10-01'
QUERY_EIGHTH = 'ibukota pindah lang:id since:2024-01-01 until:2024-10-01'
QUERY_NINTH = 'pemindahan ibukota lang:id since:2024-01-01 until:2024-10-01'


FILE_NAME_FIRST = 'ikn.csv'
FILE_NAME_SECOND = 'ibu_kota_baru.csv'
FILE_NAME_THIRD = 'ibu_kota_nusantara.csv'
FILE_NAME_FOURTH = 'ibu_kota_pindah.csv'
FILE_NAME_FIFTH = 'pemindahan_ibu_kota.csv'
FILE_NAME_SIXTH = 'ibukota_baru.csv'
FILE_NAME_SEVENTH = 'ibukota_nusantara.csv'
FILE_NAME_EIGHTH = 'ibukota_pindah.csv'
FILE_NAME_NINTH = 'pemindahan_ibukota.csv'

CURRENT_QUERY = QUERY_NINTH
CURRENT_FILE_NAME = FILE_NAME_NINTH

async def get_tweets(tweets):
  if tweets is None:
      print(f'{datetime.now()} - Mengambil tweet...')
      tweets = await client.search_tweet(CURRENT_QUERY, product="Latest")
  else:
      wait_time = randint(5, 10)
      print(f'{datetime.now()} - Mengambil tweet selanjutnya setelah {wait_time} detik...')
      time.sleep(wait_time)
      tweets = await tweets.next()
  return tweets

with open(f'datasets/raw/{CURRENT_FILE_NAME}', 'w', newline="", encoding='utf-8') as file:
  writer = csv.writer(file)
  writer.writerow(['no', 'urls', 'user_id', 'username', 'user_display_name', 'tweet_id', 'full_text', 'created_at', 'retweet_count', 'like_count'])

user_agent = 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Mobile Safari/537.36'
client = Client(language='en-US', user_agent=user_agent)
client.load_cookies('scraper/twikit_cookies.json')

async def main():
  tweet_count = 0
  tweets = None

  while tweet_count < MIN_TWEETS:
    try:
      tweets = await get_tweets(tweets)
    except TooManyRequests as e:
      rate_limit_reset = datetime.fromtimestamp(e.rate_limit_reset)
      print(f'{datetime.now()} - Rate limit tercapai. Menunggu hingga {rate_limit_reset}...')
      wait_time = rate_limit_reset - datetime.now()
      time.sleep(wait_time.total_seconds())
      continue
    
    if not tweets:
      print(f'{datetime.now()} - Tidak ada tweet lagi.')
      break
      
    for tweet in tweets:
      tweet_count += 1
      tweet_data = [tweet_count, tweet.urls, tweet.user.id, tweet.user.screen_name, tweet.user.name, tweet.id, tweet.text, tweet.created_at, tweet.retweet_count, tweet.favorite_count]

      with open(f'datasets/raw/{CURRENT_FILE_NAME}', 'a', newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(tweet_data)

    print(f'{datetime.now()} - {tweet_count} tweet berhasil diambil.')

  print(f'{datetime.now()} - Selesai! {tweet_count} tweet berhasil diambil.')

asyncio.run(main())