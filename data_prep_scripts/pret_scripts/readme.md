# Pretraining Data Processing
### Configuration
> FFmpeg installation: ```sudo apt-get install ffmpeg```

> Installing libs to your virtual env: ```pip install -r requirements.txt```
### For Downloading Data
> Required libraries ```youtube_dl, yt_dlp, pandas, ffmpeg, tqdm```

> Usage: ```bash process_data.sh </path/to/download> <num_of_threads>```

- The </path/to/download> refers to the location where the data will be downloaded.
- The <num_of_threads> can be used to control the parallelization.

The above command will start download of all the youtube-url's for the language given, extract the audio (wav) and downsample it (to 16kHz) and name it as per the unique youtube-id. Subsequent to it, the data will be passed to VAD -> SNR -> Chunking pipeline automatically.

#### Additional Tools
### For Voiced Activity Detection Step only

> Required libraries ```webrtcvad, tqdm```

> Usage: ```python vad.py <data_read_dir> <data_write_dir> <folder_name>```

- The <data_read_dir> is the root of downloaded files which contain downloaded data in language-named-folders
- The <data_write_dir> is the location for saving the data after VAD step
- The <folder_name> refers to the names of language-named-folder for which you want to perform this VAD step.

The reason why folder_name has been kept as a seperate entity is to allow parallelization because one can process multiple folders simultaneously.

### For SNR Filtering
> Required libraries ```numpy, soundfile```

> Usage: ```python snr.py <data_path> <folder/language_name>```

- The <data_path> refers to the root path containing all the audios in language specific folders. Here it refers to the <data_write_dir> from the previous step.
- The <folder/language_name> refers to name of language_specific folder for which snr_filtering needs to be applied. The audio data that is rejected is moved in the folder "snr_rejected", which is created automatically.

### For Chunking
> Required libraries ```pydub, joblib, tqdm```

> Usage: ```python chunking.py <chunking_path>```

- All the audio files present in the <chunking_path> will be chunked and saved in the same location. The original files are removed.
