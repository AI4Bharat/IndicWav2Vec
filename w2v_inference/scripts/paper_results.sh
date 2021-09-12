
#########################################################################

#Final runs:

bash infer_auto.sh 0 "hindi_mucs" "large" "mucs_small_2" 2 -4 1024 &\
bash infer_auto.sh 0 "hindi_mucs" "large" "mucs_large_2" 2 -4 1024 &\
bash infer_auto.sh 0 "hindi_mucs" "large" "comb" 2 -4 1024 &\
bash infer_auto.sh 0 "hindi_mucs" "large" "mucs_small" 2 -4 1024 &\
bash infer_auto.sh 0 "hindi_mucs" "large" "mucs_large" 2 -4 1024 &\

bash infer_auto.sh 0 "marathi_mucs" "large" "mucs_small_2" 1 -1 1024 &\
bash infer_auto.sh 0 "marathi_mucs" "large" "mucs_large_2" 1 -1 1024 &\
bash infer_auto.sh 0 "marathi_mucs" "large" "comb" 1 -1 1024 &\
bash infer_auto.sh 0 "marathi_mucs" "large" "mucs_small" 1 -1 1024 &\
bash infer_auto.sh 0 "marathi_mucs" "large" "mucs_large" 1 -1 1024 &\

bash infer_auto.sh 0 "odia_mucs" "large" "mucs_small_2" 1 -1 1024 &\
bash infer_auto.sh 0 "odia_mucs" "large" "mucs_large_2" 1 -1 1024 &\
bash infer_auto.sh 0 "odia_mucs" "large" "comb" 1 -1 1024 &\
bash infer_auto.sh 0 "odia_mucs" "large" "mucs_small" 1 -1 1024 &\
bash infer_auto.sh 0 "odia_mucs" "large" "mucs_large" 1 -1 1024 &\

bash infer_auto.sh 0 "gujarati_mucs" "large" "mucs_small_2" 5 -1 1024 &\
bash infer_auto.sh 0 "gujarati_mucs" "large" "mucs_large_2" 5 -3 1024 &\
bash infer_auto.sh 0 "gujarati_mucs" "large" "comb" 5 -1 1024 &\
bash infer_auto.sh 0 "gujarati_mucs" "large" "mucs_small" 5 -1 1024 &\
bash infer_auto.sh 0 "gujarati_mucs" "large" "mucs_large" 5 -3 1024 &\

bash infer_auto.sh 0 "telugu_mucs" "large" "mucs_small_2" 5 -2 1024 &\
bash infer_auto.sh 0 "telugu_mucs" "large" "mucs_large_2" 4 -4 1024 &\
bash infer_auto.sh 0 "telugu_mucs" "large" "comb" 4 -4 1024 &\
bash infer_auto.sh 0 "telugu_mucs" "large" "mucs_small" 5 -2 1024 &\
bash infer_auto.sh 0 "telugu_mucs" "large" "mucs_large" 4 -4 1024 &\

bash infer_auto.sh 0 "tamil_mucs" "large" "mucs_small_2" 5 -4 1024 &\
bash infer_auto.sh 0 "tamil_mucs" "large" "mucs_large_2" 4 -4 1024 &\
bash infer_auto.sh 0 "tamil_mucs" "large" "comb" 4 -4 1024 &\
bash infer_auto.sh 0 "tamil_mucs" "large" "mucs_small" 5 -4 1024 &\
bash infer_auto.sh 0 "tamil_mucs" "large" "mucs_large" 4 -4 1024 &\


bash infer_auto.sh 0 "nepali_oslr" "large" "oslr_small" 4 -3 1024 &\
bash infer_auto.sh 0 "nepali_oslr" "large" "oslr_large" 3 -1 1024 &\
bash infer_auto.sh 0 "nepali_oslr" "large" "comb" 3 -2 1024 &\
bash infer_auto.sh 0 "nepali_oslr" "large" "oslr_small_1" 4 -3 1024 &\
bash infer_auto.sh 0 "nepali_oslr" "large" "oslr_large_1" 3 -1 1024 &\

bash infer_auto.sh 0 "bengali_oslr" "large" "oslr_small" 4 -2 1024 &\ 
bash infer_auto.sh 0 "bengali_oslr" "large" "oslr_large" 4 -2 1024 &\
bash infer_auto.sh 0 "bengali_oslr" "large" "comb" 4 -2 1024 &\
bash infer_auto.sh 0 "bengali_oslr" "large" "oslr_small_1" 4 -2 1024 &\ 
bash infer_auto.sh 0 "bengali_oslr" "large" "oslr_large_1" 4 -2 1024 &\

bash infer_auto.sh 0 "sinhala_oslr" "large" "oslr_small" 3 -1 1024 &\
bash infer_auto.sh 0 "sinhala_oslr" "large" "oslr_small_1" 3 -1 1024 &\


bash infer_auto.sh 0 "telugu_msr" "large" "msr_small" 5 -4 1024 &\
bash infer_auto.sh 0 "telugu_msr" "large" "msr_large" 4 -4 1024 &\
bash infer_auto.sh 0 "telugu_msr" "large" "comb_2" 4 -4 1024 &\

bash infer_auto.sh 0 "gujarati_msr" "large" "mucs_small_2" 5 -1 1024 &\ 
bash infer_auto.sh 0 "gujarati_msr" "large" "msr_large" 5 -3 1024 &\
bash infer_auto.sh 0 "gujarati_msr" "large" "comb_2" 5 -1 1024 &\

bash infer_auto.sh 0 "tamil_msr" "large" "msr_small" 5 -4 1024 &\
bash infer_auto.sh 0 "tamil_msr" "large" "msr_large" 4 -4 1024 &\
bash infer_auto.sh 0 "tamil_msr" "large" "comb_2" 4 -4 1024 &\


bash infer_auto.sh 0 "gujarati_msr" "large" "comb_2" 5 -1 128 &\
bash infer_auto.sh 0 "tamil_msr" "large" "comb_2" 4 -4 128 &\
bash infer_auto.sh 0 "telugu_msr" "large" "comb_2" 4 -4 128 &\

bash infer_auto.sh 0 "gujarati_msr" "msr_1hour" "comb_2" 5 -1 128 &\
bash infer_auto.sh 0 "tamil_msr" "msr_1hour" "comb_2" 4 -4 128 &\
bash infer_auto.sh 0 "telugu_msr" "msr_1hour" "comb_2" 4 -4 128 &\

bash infer_auto.sh 0 "gujarati_msr" "msr_10hour" "comb_2" 5 -1 128 &\
bash infer_auto.sh 0 "tamil_msr" "msr_10hour" "comb_2" 4 -4 128 &\
bash infer_auto.sh 0 "telugu_msr" "msr_10hour" "comb_2" 4 -4 128 &\

bash infer_auto.sh 0 "gujarati_msr" "msr_20hour" "comb_2" 5 -1 128 &\
bash infer_auto.sh 0 "tamil_msr" "msr_20hour" "comb_2" 4 -4 128 &\
bash infer_auto.sh 0 "telugu_msr" "msr_20hour" "comb_2" 4 -4 128 &\


# vk models
bash infer_auto.sh 0 "nepali_oslr" "base2" "oslr_small" 3 -2 128 &\
bash infer_auto.sh 0 "bengali_oslr" "base2" "oslr_small" 4 -2 128 &\
bash infer_auto.sh 0 "sinhala_oslr" "base2" "oslr_small" 3 -1 128 &\

