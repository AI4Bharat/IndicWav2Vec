# indic-wav2vec2

### All the data preparation scripts are present in data_prep_scripts dir. 
The sequence of data preperation pipeline involve
- Downloading the data (using ```data_scrape.py```)
- Pass the data through VAD pipeline (using ```vad.py```)
- Pass the obtained data through SNR Filtering pipeline (using ```snr_filter.py```)
- Finally chunk the data (using ```chunking.py```)

Click [here](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/data_prep_scripts) for more extended documentation on how to execute these individual steps.
 
### Training Language Model

Scripts for installing, preparing data and training language model is present in [lm_training](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/lm_training) folder.

### Inference/Evaluation

Evaluation Scripts with complete documentation are present in [w2v_inference](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/w2v_inference) folder.