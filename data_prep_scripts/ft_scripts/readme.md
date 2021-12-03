###For MUCS and MSR datset
1. Make a new directory and name it (say mucs)
2. Download and extract the data inside mucs. The data should be extracted in such a way that each folder inside should contain data for a particular language i.e in each language specific folder, it should contain train, valid and test folder and within them the audio + transcript.txt
The sample structure is given below
mucs
 |----language1(hindi)
 |        |----train
 |                |----folder_containing_train_audio_files
 |                |----transcript.txt
 |        |----valid
 |                |----folder_containing_valid_audio_files
 |                |----transcript.txt
 |        |----test
 |                |----folder_containing_test_audio_files
 |                |----transcript.txt
 |----language2(marathi)
 |        |----train
 |                |----folder_containing_train_audio_files
 |                |----transcript.txt
 |        |----valid
 |                |----folder_containing_valid_audio_files
 |                |----transcript.txt
 |        |----test
 |                |----folder_containing_test_audio_files
 |                |----transcript.txt

3. After bringing the data in given format, run the following command to create manifest
bash m_process.sh <path/to/the/root/folder/(mucs)>

Note: For datasets, that are not sampled uniformly at 16kHz, before running the above set of commands, the user may run the following command to normalize the data first.
bash normalize_sr.sh <path/to/the/folder/to/normalize> <ext|wav|mp3>

