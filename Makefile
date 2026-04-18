.PHONY: all install generate train quantize demo evaluate clean

all: install generate train quantize

install:
	pip install -r requirements.txt

generate:
	python generate_data.py

train:
	python train.py

quantize:
	python quantize.py

demo:
	streamlit run streamlit_app.py

evaluate:
	python evaluate.py

verify:
	python verify_data.py

clean:
	rm -rf models/adapter models/quantized
	rm -rf data/*.jsonl
	rm -rf __pycache__