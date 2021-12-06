### Dataset processing and Manifest creation
1. Make a new directory and name it (say mucs)
2. Download and extract the data inside mucs. The data should be extracted in such a way that each folder inside should contain data for a particular language i.e in each language specific folder, it should contain train, valid and test folder and within them the audio + transcript.txt
The sample structure is given below
mucs
 ```bash
 mucs(or msr/openslr)
    ├── hindi
    │   ├── test
    │   │   ├── audio
    │   │   └── transcript.txt
    │   ├── train
    │   │   ├── audio
    │   │   └── transcript.txt
    │   └── valid
    │       ├── audio
    │       └── transcript.txt
    └── marathi
        ├── test
        │   ├── audio
        │   └── transcript.txt
        ├── train
        │   ├── audio
        │   └── transcript.txt
        └── valid
            ├── audio
            └── transcript.txt
 ```

3. After bringing the data in given format, run the following command to create manifest

- ```bash m_process.sh <path/to/the/root/folder/(mucs)>```

The final step would result in creation manifest folder in each language specific folder which can the be used with fairseq for finetuning.

Note: For datasets, that are not sampled uniformly at 16kHz, before running the above set of commands, the user may run the following command to normalize the data first.
- ```bash normalize_sr.sh <path/to/the/folder/to/normalize> <ext|wav|mp3>```
