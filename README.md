Fraud Detection

Veri

https://www.kaggle.com/c/ieee-fraud-detection/data

Indir:
- train_transaction.csv
- train_identity.csv

Data klasorune koy

Kurulum

pip install -r requirements.txt

Calistirma

Metrikler icin:
python main.py veya bash run.sh

Api icin:
python -m uvicorn app.main:app --reload --port 8001 veya bash run_api.sh

http://127.0.0.1:8001/docs

Ornek /score body:
{"transaction_id": 3400379}

api sadece val set id kabul ediyor

akis

veri > split > feature > lgbm > anomali > context > rule > rag