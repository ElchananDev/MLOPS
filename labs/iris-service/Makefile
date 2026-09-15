.PHONY: install train test serve docker clean

install:
	python -m pip install -r requirements-dev.txt

train:
	python train.py

test:
	python -m pytest -q

serve:
	python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload

docker:
	docker build -t iris-api:learning .
	docker run --rm -p 127.0.0.1:8000:8000 iris-api:learning

clean:
	rm -rf artifacts __pycache__ .pytest_cache mlruns mlflow.db
