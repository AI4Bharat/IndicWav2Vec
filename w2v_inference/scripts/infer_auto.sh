#!/bin/bash
dir=$PWD/
parentdir="$(dirname "$dir")"

# Custom values
cuda=$1
data_name=$2

aff=(${data_name//_/ })
lang=`echo ${aff[0]}`
postfix=`echo ${aff[1]}`

ckpt=$3
cff=(${ckpt//_/ })
ckpt_type=`echo ${cff[0]}`
if_cls=`echo ${cff[-1]}`

lm_name=$4 # "" for viterbi, "comb" 
lm_wt=$5
word_score=$6
beam=$7 # 128 or 1024

data_prefix='datasets_test'
subset='valid'

# Data path
data_path=${parentdir}'/'${data_prefix}'/'${data_name}''
result_path=${parentdir}'/results/'${data_name}'_'${ckpt}''

# LM paths
lm_model_path=${parentdir}'/checkpoints/language_model/'${lm_name}'/'${lang}'/lm.binary'
lexicon_lst_path=${parentdir}'/checkpoints/language_model/'${lm_name}'/'${lang}'/lexicon.lst'

# wav2vec models
if [[ ${ckpt_type} = 'comb' ]]; then
  checkpoint_path=${parentdir}'/checkpoints/acoustic_model/'${ckpt}'/'${ckpt}'.pt'
else
  checkpoint_path=${parentdir}'/checkpoints/acoustic_model/'${ckpt}'/'${lang}'_'${ckpt}'.pt'
fi  


if [ "${lm_name}" = "" ]; then
  mkdir -p ${result_path}
  CUDA_VISIBLE_DEVICES=${cuda} python3 ../infer/infer.py ${data_path} --task audio_finetuning \
  --nbest 1 --path ${checkpoint_path} --gen-subset ${subset} --results-path ${result_path} --w2l-decoder viterbi \
  --lm-weight 0 --word-score 0 --sil-weight 0 --criterion ctc --labels ltr --max-tokens 5000000 \
  --post-process letter

  if [[ ${ckpt_type} = 'comb' ]]; then
    if [[ ${if_cls} = 'cls' ]]; then
      python3 ../infer/wer/calculate_wer.py -o ${data_path}/valid.wrd -p ${result_path}/hypo.word-${ckpt}'.pt'-${subset}.txt \
      -t ${data_path}/${subset}.tsv -s save -n ${result_path}/sentence_wise_wer.csv -e true --transl=True --lang ${lang} 
    else
      python3 ../infer/wer/calculate_wer.py -o ${data_path}/valid.wrd -p ${result_path}/hypo.word-${ckpt}'.pt'-${subset}.txt \
      -t ${data_path}/${subset}.tsv -s save -n ${result_path}/sentence_wise_wer.csv -e true 
    fi
  else
    if [[ ${if_cls} = 'cls' ]]; then
      python3 ../infer/wer/calculate_wer.py -o ${data_path}/valid.wrd -p ${result_path}/hypo.word-${lang}${ckpt_postfix}'_'${ckpt}'.pt'-${subset}.txt \
      -t ${data_path}/${subset}.tsv -s save -n ${result_path}/sentence_wise_wer.csv -e true --transl=True --lang ${lang} 
    else
      python3 ../infer/wer/calculate_wer.py -o ${data_path}/valid.wrd -p ${result_path}/hypo.word-${lang}${ckpt_postfix}'_'${ckpt}'.pt'-${subset}.txt \
      -t ${data_path}/${subset}.tsv -s save -n ${result_path}/sentence_wise_wer.csv -e true
    fi
  fi

else
  kenlm_result_path=${result_path}_$4_$5_$6_$7
  mkdir -p ${kenlm_result_path}

  CUDA_VISIBLE_DEVICES=${cuda} python3 ../infer/infer.py ${data_path} --task audio_finetuning \
  --nbest 1 --path ${checkpoint_path} --gen-subset ${subset} --results-path ${kenlm_result_path} --w2l-decoder kenlm --lm-model ${lm_model_path}\
  --lm-weight ${lm_wt} --word-score ${word_score} --sil-weight 0 --criterion ctc --labels ltr --max-tokens 500000 --lexicon ${lexicon_lst_path} \
  --post-process letter --beam ${beam}

  if [[ ${ckpt_type} = 'comb' ]]; then
    python3 ../infer/wer/calculate_wer.py -o ${data_path}/valid.wrd -p ${kenlm_result_path}/hypo.word-${ckpt}'.pt'-${subset}.txt \
    -t ${data_path}/${subset}.tsv -s save -n ${kenlm_result_path}/sentence_wise_wer.csv -e true
  else
    python3 ../infer/wer/calculate_wer.py -o ${data_path}/valid.wrd -p ${kenlm_result_path}/hypo.word-${lang}${ckpt_postfix}'_'${ckpt}'.pt'-${subset}.txt \
    -t ${data_path}/${subset}.tsv -s save -n ${kenlm_result_path}/sentence_wise_wer.csv -e true
  fi  
fi

