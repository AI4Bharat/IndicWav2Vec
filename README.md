# IndicWav2Vec2

### All the data preparation scripts are present in data_prep_scripts dir. 
The sequence of data preperation pipeline involve
- Downloading the data (using ```data_scrape.py```)
- Pass the data through VAD pipeline (using ```vad.py```)
- Pass the obtained data through SNR Filtering pipeline (using ```snr_filter.py```)
- Finally chunk the data (using ```chunking.py```)

Click [here](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/data_prep_scripts) for more extended documentation on how to execute these individual steps.

### Manifest Creation

For creating language-wise pretraining manifest
``` shell script
$ python path/to/lang_wise_manifest_creation.py /path/to/wave/files --dest /manifest/path --ext $ext --valid-percent $valid
```

For ```/path/to/wav/files/``` we expect the directory to have one folder per language under the parent directory

In our pretraing, we use a ```--valid-percent``` as 0.03

For creating a combined validation file for all languages, we concatenate all individual ```*_valid.tsv``` files to create a ```valid.tsv``` file.

```
import pandas as pd
import glob

filenames = glob.glob("*_valid.tsv")

combined = []
for f in filename:
    df = pd.read_csv(f, skiprows=1, names=['f', 'd'], sep='\t')
    combined.append(df)

df_combined = pd.concat(combined, axis=0, ignore_index=True)
df_combined.to_csv('valid.tsv', index=True, header=False, sep='\t')
```

We then add the ```/path/to/wav/files/```  to the first line of the valid.tsv file

### Pretraining

For pretraining the model we do multi-node training and schedule the runs with slurm.

Following is the invocation script for training IndicWav2Vec base starting from Wav2Vec2.0 English base ckeckpoint
``` shell script
$ sbatch --job-name <NAME> --gres gpu:<N_GPU_PER_NODE> --cpus-per-task <N_CPUS> \
    --nodes <N_NODES> --ntasks-per-node <N_TASKS> \
    --wrap "srun --output train.log.node%t --error train.stderr.node%t.%j \
        $(which fairseq-hydra-train) \
        task.data=/path/to/manifest/directory \
        common.wandb_project=<wandb project name> \
        task._name=temp_sampled_audio_pretraining \
        +task.sampling_alpha=0.7 \
        common.log_interval=200 \
        common.log_format=tqdm \
        dataset.max_tokens=3000000 \
        common.user_dir=/path/to/custom_task/directory \
        checkpoint.save_dir=/path/to/save/model/checkpoints \
        checkpoint.restore_file=/path/to wav2vec2-english-base/checkpoint.pt \
        +optimization.update_freq='[2]' \
        optimization.clip_norm=0.5 \
        checkpoint.reset_optimizer=true \
        distributed_training.distributed_world_size=<total GPUs> \
        distributed_training.distributed_port=$PORT \
        --config-dir /path/to/configs/directory \
        --config-name wav2vec2_base_librispeech"
```

For Large model we override the above configuration with 
```
checkpoint.restore_file=/path/to wav2vec2-english-large/checkpoint.pt \
+optimization.update_freq='[6]' \
lr_scheduler.warmup_updates=0 \
--config-name wav2vec2_large_librivox"
```

Configs for both the models are provided in the [configs](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/configs) directory

### Fine-tuning process

#### Manifest creation
We use fairseq's standard ```wav2vec_manifest.py``` for creating tsv files from audio data files. The same can be done using the following command after installing fairseq
> ```python <fairseq_path>/examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext $ext --valid-percent $valid```

After that the labels for each 'tsv' files can be created by appropriately making '.wrd' files which contain label (transcription) for the audio file present in the same relative line of 'tsv' file. In addition to that '.ltr' files are formed from '.wrd' files in which individuals characters are spaced out and a '|' is inserted between the two adjacent words and also at the end.
e.g 'HELLO WORLD' -> 'H E L L O | W O R L D |'

The sample scripts are already inluded in the data_prep_scripts folder (```msr_data_post_manifest.py msr_data_post_manifest.py, mucs_post_manifest.py, openslr_post_manifest.py```) for MSR, MUCS and OpenSLR data respectively. These scripts have to be run individually for each language in the dataset with appropriate path changes. 

#### Fine-tune

Following is the invocation script for finetuning IndicWav2Vec large on a particular language
    
```
sbatch --job-name <NAME> --gres gpu:<N_GPU_PER_NODE> --cpus-per-task <N_CPUS> \
    --nodes <N_NODES> --ntasks-per-node <N_TASKS> \
    --wrap "srun --output finetune.log.node%t --error finetune.stderr.node%t.%j \
        $(which fairseq-hydra-train) \
        task.data=/path/to/finetune/manifest/directory/for/a/particular/language \
        common.wandb_project=<wandb project name> \
        model.w2v_path=/path/to/pretrained/model_large.pt \
        common.log_interval=50 \
        common.log_format=tqdm \
        dataset.max_tokens=1000000 \
        checkpoint.save_dir=/path/to/save/model/fine_tune_checkpoints \
        +optimization.update_freq='[1]' \
        distributed_training.distributed_world_size=<total GPUs> \
        --config-dir /path/to/configs/directory \
        --config-name ai4b_xlsr"
```

For IndicWav2Vec Base model we override the above configuration with 
```
model.w2v_path=/path/to/pretrained/model_base.pt \
--config-name ai4b_base"
```

Configs for both the models are provided in the [finetune_configs](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/finetune_configs) directory

### Training Language Model

Scripts for installing, preparing data and training language model is present in [lm_training](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/lm_training) folder.

### Inference/Evaluation

Evaluation Scripts with complete documentation are present in [w2v_inference](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/w2v_inference) folder.
