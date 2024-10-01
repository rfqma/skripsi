# Analisis Reaksi Publik Terhadap Pemindahan Ibu Kota Negara Menggunakan Metode Lexicon Based dan KNN

> Analyze opinions expressed on X (formerly Twitter) regarding the relocation of Indonesia's capital city using combination of algorithm classifiers K-Nearest Neighbor (KNN), Feature Extraction Term Frequency-Inverse Document Frequency (TF-IDF), Data Balancing (SMOTE), Feature Selection (Chi-Square) and also using a Lexicon-based approach for labeling data as positive, negative, or netral sentiment.

# Input

- `dataset`
  - Data regarding the relocation of Indonesia's capital city from X (formerly Twitter).
  - Data crawling was done at 21 August 2024 with keywords: _ibu_kota_baru_, _ibu_kota_nusantara_, _ibu_kota_pindah_, _ikn_, _pemindahan_ibu_kota_.
  - Some X Search Queries like: _lang:id_, _since:2023-01-01_.
  - Also the crawler use `LATEST` tab from X.
- `slang, stopwords, negation, etc`
  - [/dictionaries](https://github.com/rfqma/skripsi/tree/master/dictionaries)
- `Custom Lexicon`
  - [/dictionaries/lexicon](https://github.com/rfqma/skripsi/tree/master/dictionaries/lexicon)
- `Ekstraksi Fitur`
  - `TF-IDF`
- `Data Balancer`
  - `SMOTE`
- `Feature Selection`
  - `Chi-Square`
- `Classifier Algorithm`
  - `K-Nearest Neighbor (KNN)`
- `Evaluation`
  - `Confusion Matrix`
  - `Classification Report`

# Dependencies

> Look at `requirements.txt` for more details.

# Flowchart

> _`coming soon`_

# Getting Started

## Cookies

> - Export x cookies from browser with some cookies extractor extension on browser.
> - Save it to _`scraper/raw_cookies.json`_.
> - Generate _`twikit`_ cookies with _`scraper/cookies.py`_.
> - Save it to _`scraper/twikit_cookies.json`_.
> - Run _`scraper/scraper.py`_.

## Prerequisites On Machine

- `python`
- `pip`
- `git`
- `ipykernel`

## Setup

- Install dependencies

```bash
pip install -r requirements.txt
```

- Check installed dependencies on machine

```bash
pip freeze
```

- Usage Order

```
1. scraper/cookies.py
2. scraper/scraper.py
3. [ind|eng]-preprocess.ipynb
4. [ind|eng]-sentiment.ipynb
5. [ind|eng]-knn-[inset|sentistrength].ipynb
6. [ind|eng]-streamlit-[inset|sentistrength].py
```

- run streamlit

```bash
streamlit run [ind|eng]-streamlit-[inset|sentistrength].py
```

# NOTES!!

> [!WARNING]
> googletrans==3.1.0a0 use older version of httpx, which is not compatible with twikit==2.1.2, twikit use httpx latest version.
