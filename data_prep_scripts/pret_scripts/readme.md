# Pretraining Data Processing

### For Downloading and Processing YT Data
> Required libraries ```youtube_dl, yt_dlp, pandas, ffmpeg, tqdm```

> Usage: ```bash process_data.sh </path/to/download> <num_of_threads>```

- The </path/to/download> refers to the location where the data will be downloaded.
- The <num_of_threads> can be used to control the parallelization.

The above command will start download of all the youtube-url's for the language given, extract the audio (wav) and downsample it (to 16kHz) and name it as per the unique youtube-id. Subsequent to it, the data will be passed to VAD -> SNR -> Chunking pipeline automatically.

### For Downloading and Processing NoA Data
> Required libraries ```ffmpeg, tqdm```
1. Download the NoA from the publicly availiable links
2. Put the data in language specific folders
3. Run ```bash normalize_sr.sh <path/to/root/of/NoA>``` to normalize the SR and number of channels
4. Run ```python vad.py <path/to/root/of/NoA> <path/to/refined/data/storage> language-specific-foldername ```
5. Run ```python snr_filter.py <path/to/refined/data/storage> language-specific-foldername <path/to/store/rejected/files>```
5. Run ```python chunking.py <path/to/refined/data/storage/languagespecificfolder>```

- The <path/to/root/of/NoA> root path to NoA directory.

### For Processing Individual Directories
1. Download the data using ```bash dw_util.sh <path/to/txt/of/a/particular/language> <path/to/root/where/data/will/be/stored> <#ofthreads>
2. Pass the data through VAD step as given below
3. Pass the data through SNR setp as given below
4. Pass the data through Chunking as given below


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
