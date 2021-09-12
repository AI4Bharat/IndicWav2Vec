dir=$PWD/
parentdir="$(dirname "$dir")"

### Values to change,if any - start ###
lm_name=$1
top_k=2000000
lexicon=2000000

#For kenlm
lm_path=${parentdir}"/lm/"${lm_name}
kenlm_bins=${parentdir}"kenlm/build/bin"
input_txt_file_path=${lm_path}"/clean_toks.txt"

#For lexicon 
lexicon_vocab_file=${lm_path}"/lexicon.txt"
path_to_save_lexicon=${out_dir}"/"${lm_name}"/lexicon.lst"

printf "\n** Generating kenlm **\n"
python ../utils/train_lm.py --input_txt ${input_txt_file_path} --lm_dir ${lm_path} \
    --lexicon ${lexicon} --top_k ${top_k} --kenlm_bins ${kenlm_bins} \
    --arpa_order 6 --max_arpa_memory "95%" --arpa_prune "0|0|0|0|1|2" \
    --binary_a_bits 255 --binary_q_bits 8 --binary_type trie 
printf "**Kenlm Generated at : "${lm_path}

