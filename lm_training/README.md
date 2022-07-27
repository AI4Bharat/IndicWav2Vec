# LM Training

## Setup

> Install KenLM and its dependencies
```
sudo apt install build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev
```
```
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build && cd build
cmake .. 
make -j 16
cd ../..
```

> Install python libraries:  ```pip install -r requirements.txt```

## Data Preparation
- 
- 
-  
- 

## Usage
> Combine Text Data and Create Lexicon: 
```python utils/clean_corpus.py -d=<lm directory path> -l=<lang> --transcript=<speech transcript folder path> --st=<start code of lang> --en=<end code of lang> --top_k=<'k' most frequent words for vocab>```

> Run lm-training: ```bash scripts/train_lm.sh <lm directory path> <lang>``` . Ouput will be generate at the same location ```"<lm directory path>/<lang>"```.

## Test Example
```
# Prepare lm_data and lexicon
python utils/clean_corpus.py -d=test_dataset/ -l hindi --st=2304 --en=2431 --transcript=test_dataset/hindi/

# Run training
bash scripts/train_lm.sh test_dataset/ hindi
```