# Penerapan Metode Klasifikasi KNN dan Lexicon Based dengan Ekstraksi Fitur fastText untuk Analisis Sentimen Publik Terhadap Pemindahan Ibu Kota Negara

> Analyze opinions expressed on X (formerly Twitter) regarding the relocation of Indonesia's capital city using combination of algorithm classifiers K-Nearest Neighbor (KNN), Feature Extraction fastText, SMOTE for balancer, and also Lexicon-based approach for labeling data as positive, negative, or netral sentiment.

# Input

- `dataset`
  - Data regarding the relocation of Indonesia's capital city from X (formerly Twitter).
  - Data crawling was done at 25 October 2024 with keywords: _ibu_kota_baru_, _ibu_kota_nusantara_, _ibu_kota_pindah_, _ikn_, _pemindahan_ibu_kota_.
  - Some X Search Queries like: _lang:id_, _since:2024-01-01_, _until:2024-10-01_.
  - Also the crawler use `LATEST` or `TOP` tab from X.
- `slang, stopwords, negation, etc`
  - [/dictionaries](https://github.com/rfqma/skripsi/tree/master/dictionaries)
- `Custom Lexicon`
  - [/lexicons](https://github.com/rfqma/skripsi/tree/master/lexicons)
- `Ekstraksi Fitur`
  - `fastText`
- `Balancer`
  - `SMOTE`
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

## Prerequisites On Machine

- `git`
- `python`
- `pip`
- `virtualenv`

## Cookies

> - Export x cookies from browser with some cookies extractor extension on browser.
> - Save it to _`scraper/raw_cookies.json`_.
> - Generate _`twikit`_ cookies with _`scraper/cookies.py`_.
> - Save it to _`scraper/twikit_cookies.json`_.
> - Run _`scraper/scraper.py`_.

## fastText

> read more on [/models/fastText](https://github.com/rfqma/skripsi/tree/master/models/fastText)

## Setup

- Initialize _`virtualenv`_

```bash
virtualenv <virtualenv_name>
```

- Activate _`virtualenv`_

  - _`Windows`_
    ```bash
    <virtualenv_name>/Scripts/activate
    ```
  - _`Linux/macOS`_
    ```bash
    source <virtualenv_name>/Scripts/activate
    ```

- Install dependencies on _`virtualenv`_

```bash
pip install -r requirements.txt
```

- Check installed dependencies on _`virtualenv`_

```bash
pip freeze
```

- Install a new kernel for _`Jupyter Notebook`_ named _`<virtualenv_name>`_ on _`virtualenv`_

```bash
python -m ipykernel install --user --name <virtualenv_name>
```

- Update _`pip`_

```bash
python -m pip install wheel setuptools pip --upgrade
```

- Start _`Jupyter Notebook`_ server on _`virtualenv`_

```bash
jupyter notebook
```

- Deactivate _`virtualenv`_

```bash
deactivate
```

## Usage Order

```bash
1. scraper/cookies.py
2. scraper/scraper.py
3. preprocess.ipynb
4. sentiment.ipynb
5. knn-[7030|8020|9010].ipynb
6. model_test.ipynb
6. streamlit.py
```

## Streamlit

```bash
streamlit run streamlit.py
```

<!-- # NOTES!!

> [!WARNING]
> googletrans==3.1.0a0 use older version of httpx, which is not compatible with twikit==2.1.2, twikit use httpx latest version. -->
