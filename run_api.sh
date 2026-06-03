set -e
cd "$(dirname "$0")"

if [ ! -f data/train_transaction.csv ] || [ ! -f data/train_identity.csv ]; then
  echo "data/ icinde train_transaction.csv ve train_identity.csv lazim"
  exit 1
fi

python -m uvicorn app.main:app --reload --port 8001
