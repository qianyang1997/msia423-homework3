image:
	docker build -t cloud .

acquire:
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ cloud run.py acquire --config=config/config.yaml

data/cloud.csv: config/test.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ cloud run.py read --config=config/config.yaml --output=data/cloud.csv

read: data/cloud.csv

tests/data/featurized.csv: data/cloud.csv config/config.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ cloud run.py featurize --input=data/cloud.csv --config=config/config.yaml --output=tests/data/featurized.csv

featurize: tests/data/featurized.csv

tests/result/predictions.csv: tests/data/featurized.csv config/config.yaml
    docker run --mount type=bind,source="$(shell pwd)",target=/app/ cloud run.py train --input=tests/data/featurized.csv --config=config/config.yaml --output=tests/result/predictions.csv

evaluate:
    docker run --mount type=bind,source="$(shell pwd)",target=/app/ cloud run.py evaluate --input=tests/result/predictions.csv

tests:
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ cloud pytest tests

clean:
	rm tests/data/* tests/result/*

all: data/raw.csv data/clean.csv data/features.csv

.PHONY: tests clean 