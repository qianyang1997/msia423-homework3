image:
	docker build -t cloud .

data/cloud.data: config/config.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ cloud run.py acquire --config=config/config.yaml --output=data/cloud.data

acquire: data/cloud.data

data/cloud.csv: data/cloud.data config/config.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ cloud run.py read --input=data/cloud.data --config=config/config.yaml --output=data/cloud.csv

read: data/cloud.csv

data/featurized.csv: data/cloud.csv config/config.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ cloud run.py featurize --input=data/cloud.csv --config=config/config.yaml --output=data/featurized.csv

featurize: data/featurized.csv

data/predictions.csv: data/featurized.csv config/config.yaml
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ cloud run.py train --input=data/featurized.csv --config=config/config.yaml --output=data/predictions.csv

train: data/predictions.csv

evaluate:
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ cloud run.py evaluate --input=data/predictions.csv

tests:
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ --entrypoint "pytest" cloud tests

clean:
	rm data/*

all: acquire read featurize train evaluate

.PHONY: tests clean image acquire read featurize evaluate train all