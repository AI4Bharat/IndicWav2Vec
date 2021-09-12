# IndicWav2Vec2

### All the data preparation scripts are present in data_prep_scripts dir. 
The sequence of data preperation pipeline involve
- Downloading the data (using ```data_scrape.py```)
- Pass the data through VAD pipeline (using ```vad.py```)
- Pass the obtained data through SNR Filtering pipeline (using ```snr_filter.py```)
- Finally chunk the data (using ```chunking.py```)

Click [here](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/data_prep_scripts) for more extended documentation on how to execute these individual steps.

### Manifest Creation

### Pretraining

For pretraining the model we do multi-node training and schedule the runs with slurm.

Following is the invocation script for training IndicWav2Vec base starting from Wav2Vec2.0 English base ckeckpoint
```
sbatch --job-name <NAME> --gres gpu:<N_GPU_PER_NODE> --cpus-per-task <N_CPUS> \
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

### Fine-tuning


### Training Language Model

Scripts for installing, preparing data and training language model is present in [lm_training](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/lm_training) folder.

### Inference/Evaluation

Evaluation Scripts with complete documentation are present in [w2v_inference](https://github.com/AI4Bharat/indic-wav2vec2/tree/main/w2v_inference) folder.