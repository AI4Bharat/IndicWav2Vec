# Running Evaluation

## Setup

### Using Docker Image (Recommended)

> Create Docker image:  ```docker build -t ai4bharat/indicw2v .```  
> Run Docker container: 
```
docker run --gpus all -it --rm -v <checkpoints_&_dataset_location>:/workspace ai4bharat/indicw2v
```

### From Source

> Install Linux dependencies: 
```
sudo apt-get install liblzma-dev libbz2-dev libzstd-dev libsndfile1-dev libopenblas-dev libfftw3-dev libgflags-dev libgoogle-glog-dev build-essential cmake libboost-system-dev libboost-thread-dev libboost-program-options-dev libboost-test-dev libeigen3-dev zlib1g-dev libbz2-dev liblzma-dev
```

> Install KenLM
```
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir -p build && cd build
cmake .. 
make -j 16
cd ..
export KENLM_ROOT=$PWD
cd ..
```

> Install python libraries:  ```pip install -r requirements.txt```

> Install torch from official repo: [PyTorch Official](https://pytorch.org/get-started/locally/)

> Install fairseq: 
```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
```
> Install Flashlight:

```
git clone https://github.com/flashlight/flashlight.git
cd flashlight/bindings/python
export USE_MKL=0
python setup.py install
cd ../../..
```

## Data Preparation
- Make dataset and checkpoint directories ```mkdir datasets_test && mkdir checkpoints && mkdir checkpoints/language_model && mkdir checkpoints/acoustic_model```
- Prepare test manifest folder using [fairseq](https://github.com/pytorch/fairseq/tree/master/examples/wav2vec) and put the manifest folder inside ```datasets_test``` folder. The ```<data_folder_name>``` must be of the form, ```<lang>_*```, where ```lang``` can be hindi, bengali, gujarati, tamil, telugu, tamil, nepali, sinhala and odia.
- Download/Train fine-tuning and language model checkpoints and put it inside ```checkpoints/acoustic_model``` and ```checkpoints/language_model``` folder respectively. Note: ```<am_folder_name>``` must contain checkpoints whose name should be of the form: ```<lang>_<am_folder_name>.pt``` and ```<lm_folder_name>``` must contain folder ```<lang>``` with ```lm.binary``` and ```lexicon.lst```.

## Usage
> Run inference: 
```
cd scripts
bash infer_auto.sh <cuda_device_no> <data_folder_name> <am_folder_name> <lm_folder_name> <lm_weight> <word_score> <beam_width>
```
> Scripts to be run to reproduce results from **paper** are provided [here](https://github.com/AI4Bharat/indic-wav2vec2/blob/main/w2v_inference/scripts/paper_results.sh).
