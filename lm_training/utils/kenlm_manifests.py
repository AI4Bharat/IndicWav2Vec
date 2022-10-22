# %% [markdown]
# # Text Cleaning for KenLM Training
# 

# %% [markdown]
# Import Statements

# %%
LANG = ''

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize.sentence_tokenize import sentence_split
from num_to_words.num_to_words import num_to_word
from joblib import Parallel, delayed
from operator import itemgetter
from matplotlib import pyplot as plt
from collections import Counter

import shutil
import regex
import os
import string
from tqdm import tqdm
import pandas as pd
import re
import itertools
import time
import argparse

# %%
## TODO: List of experiments
'''
1. Tasks: 
"For all the experiments below, make parallels plot for both efficiency and accuracy(wer)" (as in wandb).
"Also for the most accurate models and most efficiet models, plot efficiency vs accuracy".
    1. Run following ablations: 
        a. Indic-corp-v5 vs Indic-corp-v4 [Low Priority, default is "indic-corp-v5"]
        b. Clean, Small text-data vs Noisy, Large text-data [To be Done as part of task#2]
        c. Effect of n in n-grams [To be Done as part of task#2]
        d. Effect of pruning in n-grams [Low Priority, default is "do pruning"]
        e. Effect of filtering in n-grams [To be Done as part of task#3]
        f. Effect of decoder type and hyperparameters like alpha, beta
    2. [Kenlm ablations] Train lms on:
        a. Clean, Small filtered text-data vs Noisy, Large unfiltered text-data*
        b. Small(3gram) vs Large(6gram) LMs 
        c. Take Vakyansh's LM as-it-is
    3. [Lexicon ablations]
        Filter the above lms accrding to following filters:
        a. Full lexicon, No filtering
        b. Total top-k words lexicon filtering
        c. Only top-k words from indic-corp filtering plus all words from transcripts
        d. Take Vakyansh's Lexicon as-it-is
    4. [Decoder ablations]:
        a. Flashlight lexicon-based decoding:
            a. Ablation on hyperparameters like beam-size, beam-threshold, beam-size-token, alpha and beta.
        b. pyctcdecode:
            a. Ablation on hyperparameters like beam-size, apha and beta.
    
2. Datasets we need: indic-corp-v5, indic-corp-v4, all speech training transcripts (MUCS/MSR, DC, NoA, etc.)
3. Cleaning and filtering guidelines:
    1. Remove sents from train-set those are there in the test-set 
    2. Normalize using indic-nlp-library
    3. For clean set: 
        0. Create Dict file (contains "curated" language chars, without digits)
        1. Do num2word for numbers whose length <=4
        2. Create Foreign characters list (all except "curated" language chars, standard puncts, currency symbols and spaces)
        3. Drop sents which have any foreign characters.
        4. Only select those sents with num_words in [min_words, max_words] range, with num_non_lang_chars in [min, max], 
        all calculated using population stats
        5. Drop all the characters which are not in the Dict.
    4. For noisy set: 
        0. Create Dict file (contains "all" language chars, all english chars and apostrophe)
        1. Create Foreign characters list (all except "all" language chars, all english chars, all puncts, currency symbols and spaces)
        2. 
'''

# %% [markdown]
# ## [KenLM#1] 2.clean-text.large._.3c._.4a
# 

# %%
# Steps to do (Creating clean set):
'''
For indic-corp:
1. Import shuffled paragraphs. 
2. Normalize and do num2word (if available). Also remove the bracketed contents, using regex.
3. Create Allowed character list (all dict chars + puncts + english chars)
4. Check how many foreign characters (not in allowed chars) are there in the paragraph. Create a distribution across paragraphs. 
Select a threshold using density estimation. If it is greater than a threshold (~0.05), reject the paragraph.
6. Drop all those chars which are not in dict.

For train-transcripts:
1. Import the transcripts.
2. Normalize and do num2word (if available)
3. Check how many foreign characters (not in dict) are there in the paragraph. Create a distribution across paragraphs. Select a threshold using density estimation. If it is greater than a threshold (~0.05), reject the paragraph.
4. Drop all those chars which are not in dict.
'''

# %% [markdown]
# Config and global Variables

# %%
LANG_LIST = ["hindi", "marathi", "sanskrit", "bengali", "punjabi", "gujarati", "odia", "tamil", "telugu", "kannada", "malayalam", "sinhala"]

DIGITS_SCRIPT = {
    'Devanagari': ['०', '१', '२', '३', '४', '५', '६', '७', '८', '९'],
    'Bengali': ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯'],
    'Gurmukhi': ['੦', '੧', '੨', '੩', '੪', '੫', '੬', '੭', '੮', '੯'],
    'Gujarati': ['૦', '૧', '૨', '૩', '૪', '૫', '૬', '૭', '૮', '૯'],
    'Odia': ['୦', '୧', '୨', '୩', '୪', '୫', '୬', '୭', '୮', '୯'],
    'Tamil': ['௧', '௨', '௩', '௪', '௫', '௬', '௭', '௮', '௯', '௰', '௱', '௲'],
    'Telugu': ['౦', '౧', '౨', '౩', '౪', '౫', '౬', '౭', '౮', '౯'],
    'Malayalam': ['൦', '൧', '൨', '൩', '൪', '൫', '൬', '൭', '൮', '൯'],
    'Kannada': ['೦', '೧', '೨', '೩', '೪', '೫', '೬', '೭', '೮', '೯'],
    'Arabic': ['٠' ,'١' ,'٢' ,'٣' ,'٤' ,'٥' ,'٦' ,'٧' ,'٨' ,'٩']
}
CODE2LANG = {"bn": "bengali", "gu": "gujarati", "hi": "hindi", "kn": "kannada", "ml": "malayalam",\
     "mr": "marathi", "or": "odia", "pa": "punjabi", "sa": "sanskrit", "ta": "tamil", "te": "telugu", "ur": "urdu" } 

LANG2CODE = dict((v, k) for k, v in CODE2LANG.items())

LANG2SCRIPT = {
    "hindi": "Devanagari",
    "marathi": "Devanagari",
    "sanskrit": "Devanagari",
    "bengali": "Bengali",
    "punjabi": "Gurmukhi",
    "gujarati": "Gujarati",
    "odia": "Odia",
    "tamil": "Tamil",
    "telugu": "Telugu",
    "kannada": "Kannada",
    "malayalam": "Malayalam",
    "sinhala": "Sinhala",
    "urdu": "Arabic"
}

SCRIPT2LANG = dict((v, k) for k, v in LANG2SCRIPT.items())

dir_path = "/nlsasfs/home/ai4bharat/manidl/speechteam/superb_kenlms"
indiccorp_path = "/nlsasfs/home/ai4bharat/manidl/speechteam/indic-corp-jun-2022"
dc_path = "/nlsasfs/home/ai4bharat/manidl/speechteam/dc_manifests"
superb_dict_path = "/nlsasfs/home/ai4bharat/manidl/speechteam/superb_dicts"

UNICODE_RANGE = {
    "Devanagari": (2304,2431),
	"Bengali": (2432,2559),
	"Gurmukhi": (2560,2687),
	"Gujarati": (2688,2815),
	"Odia": (2816,2943),
	"Tamil": (2944,3071),
	"Telugu": (3072,3199),
	"Kannada": (3200,3327),
	"Malayalam": (3328,3455),
	"Sinhala": (3456,3583),
    "Arabic": (1536,1791)
}   
# ref: https://www.ssec.wisc.edu/~tomw/java/unicode.html

# %% [markdown]
# LOCAL VARIABLES

# %%
if LANG != '':
    LANG_CODE = LANG2CODE[LANG]
    os.makedirs(f'{dir_path}/{LANG}', exist_ok=True)
    dict_path = f'{superb_dict_path}/{LANG}_dict.txt'
    # corpus_path = f'{indiccorp_path}/{lang_code}.txt.shuf'
    corpus_path = f'{indiccorp_path}/{LANG_CODE}_sents.txt'
    df_corpus_path = f'{dir_path}/{LANG}/clean_and_stats.tsv'
    lexicon_path = f'{dir_path}/{LANG}/lexicon.txt'
    plots_dir = f'{dir_path}/{LANG}/plots'
    clean_corpus_path = f'{dir_path}/{LANG}/cleaned_sents.txt'
    stats_path = f'{dir_path}/{LANG}/{plots_dir}/stats.tsv'
    os.makedirs(plots_dir, exist_ok=True)

# %% [markdown]
# Functions

# %%
# Normalize text
def normalize_sentence(sentence, lang_code):
    '''
    Perform NFC -> NFD normalization for a sentence and a given language
    sentence: string
    lang_code: language code in ISO format
    '''
    factory=IndicNormalizerFactory()
    normalizer=factory.get_normalizer(lang_code)
    normalized_sentence = normalizer.normalize(sentence)
    return normalized_sentence

# %%
# Create allowed char list
def allowed_chars(dict_path, lang, script_start=None, script_end=None, punc_start=8192, punc_end=8334):
    '''
    Returns dict of all allowed chars as defined below,
    "valid" are the language chars as defined in the dict_path, 
    "extras" are the extra chars in the corresponding script,
    "punc" contains all punctuations, 
    "english" include english chars

    dict_path: path to the dictionary file (containing all lang chars ('\n' delimited))
    script_start/script_end: Range of characters from script taken from [here](https://www.ssec.wisc.edu/~tomw/java/unicode.html)
    punc_start/punc_end: Range of punctuations taken from [here](https://www.ssec.wisc.edu/~tomw/java/unicode.html)
    '''
    dictionary = open(dict_path).readlines()[1:] # Dont take the first character as it is '|'
    superb_chars = [d.strip() for d in dictionary if d.strip() != ''] + [" "] # Add space to the list of valid chars

    extra_chars = None
    if script_start is None or script_end is None:
        try:
            script_start, script_end = UNICODE_RANGE[LANG2SCRIPT[lang]]
            all_chars = [chr(i) for i in range(script_start, script_end+1)] + ['₹']
            extra_chars = set(all_chars) - set(superb_chars)
        except Exception as e:
            print(e)

    puncs = [chr(i) for i in range(punc_start, punc_end+1)] + list(string.punctuation) + ["॥"] + ["।"] + ["۔"] #include all language puncs
    english = [chr(i) for i in range(128)] # Range of basic-latin [ref](https://www.ssec.wisc.edu/~tomw/java/unicode.html)

    allowed_char_dict = {"valid": set(superb_chars), "extras": extra_chars, "english": set(english), "punc": set(puncs)}
    return allowed_char_dict

# %%
def ssplit_paragraphs(input, lang):
    lang_code = LANG2CODE[lang]
    return sentence_split(input, lang_code)

# %%
def get_sents(corpus_path, lang):
    print('Loading Data')
    corpus = open(corpus_path).readlines()
    print('Done Loading Data')
    ssplit_list = Parallel(n_jobs=-1)(delayed(ssplit_paragraphs)(p, lang) for p in tqdm(corpus))
    return list(itertools.chain.from_iterable(ssplit_list))

# %% [markdown]
# Test do_num2word function
# 

# %%
## NOTE: TESTING SCRIPT

# w = '120एकक[1]दम12'
w = "abhi"
lang_code = 'hi'
set_digits = set(DIGITS_SCRIPT[LANG2SCRIPT[CODE2LANG[lang_code]]] + list(string.digits))

# Test num2word
def do_num2word(w):
    set_word = set(w)
    print(set_digits - set_word)
    print(set_word - set_digits)
    if len(set_digits - set_word) < len(set_digits):
        if len(set_word - set_digits) == 0:
            word_num = num_to_word(w, lang_code)
        else:
            str_list = []
            tmp_str = ''
            is_curr_char = False
            is_curr_digit = False
            for c in w:
                if c in set_digits:
                    if is_curr_char:
                        str_list.append(tmp_str)
                        tmp_str = ''
                    is_curr_digit = True
                    is_curr_char = False
                    tmp_str += c
                else:
                    if is_curr_digit:
                        str_list.append(num_to_word(tmp_str, lang_code))
                        tmp_str = ''
                    is_curr_digit = False
                    is_curr_char = True
                    tmp_str += c
            if is_curr_digit:
                str_list.append(num_to_word(tmp_str, lang_code))
            if is_curr_char:
                str_list.append(tmp_str)
            
            word_num = " ".join(str_list)
    else:
        word_num = w
    return word_num
# print(do_num2word(w))

# %%
def clean_and_stats(input, lang, allowed_chars_dict, set_digits, ssplit=False, remove_bracketed_content=True, remove_punctuation=True, normalize=True):
    '''
    Stats to return are, length, num_words, allowed_char_stats, other_stats
    '''
    lang_code = LANG2CODE[lang]

    # Remove bracketed content
    no_bracs = re.sub("[\(\[].*?[\)\]]", "", input) if remove_bracketed_content else input
    # no_puncs = re.sub(r'[^\w\s]','',no_bracs)

    # Remove punctuations TODO: If the logic also includes indian puncts
    remove = regex.compile(r'[\p{C}|\p{P}|\p{S}|\p{Z}]+', regex.UNICODE)
    no_puncs = remove.sub(" ", no_bracs).strip() if remove_punctuation else no_bracs

    # Normalize and num2word
    norm_input = normalize_sentence(no_puncs, lang_code) if normalize else no_puncs
    # norm_input = no_puncs # Only for Urdu
    words_list = norm_input.split()
  
    def do_num2word(w):
        num2word_count = 0
        set_word = set(w)
        if len(set_digits - set_word) < len(set_digits):
            if len(set_word - set_digits) == 0:
                try:
                    word_num = num_to_word(w, lang_code)
                    num2word_count += 1
                except:
                    word_num = w
                    print(f"Error in num2word for '{w}'")
            else:
                str_list = []
                tmp_str = ''
                is_curr_char = False
                is_curr_digit = False
                for c in w:
                    if c in set_digits:
                        if is_curr_char:
                            str_list.append(tmp_str)
                            tmp_str = ''
                        is_curr_digit = True
                        is_curr_char = False
                        tmp_str += c
                    else:
                        if is_curr_digit:
                            try:
                                str_list.append(num_to_word(tmp_str, lang_code))
                                num2word_count += 1
                            except:
                                str_list.append(tmp_str)
                                print(f"Error in num2word for '{w}'")
                            
                            tmp_str = ''
                        is_curr_digit = False
                        is_curr_char = True
                        tmp_str += c
                if is_curr_digit:
                    try:
                        str_list.append(num_to_word(tmp_str, lang_code))
                        num2word_count += 1
                    except:
                        str_list.append(tmp_str)
                        print(f"Error in num2word for '{w}'")
                if is_curr_char:
                    str_list.append(tmp_str)
                
                word_num = " ".join(str_list)
        else:
            word_num = w
        return [word_num, num2word_count]
        
    count_num2words = 0
    if lang_code not in ['ml', 'ur', 'sa']:
        # num2word_out_list = Parallel(n_jobs=32)(delayed(do_num2word)(w) for w in words_list)
        num2word_out_list = [do_num2word(w) for w in words_list]
        # modified_words_list, count_num2words = map(list, zip(*num2word_out_list))
        # code to split it into 2 lists
        modified_words_list = list(map(itemgetter(0), num2word_out_list))
        count_num2words = sum(list(map(itemgetter(1), num2word_out_list)))

    else:
        modified_words_list = words_list

    norm_out = " ".join(modified_words_list)

    # Sentence Split
    if ssplit:
        sentences = sentence_split(norm_out, lang_code)
    else:
        sentences = norm_out

    # Calculate Stats
    def stats_word(w):
        punc, extra, english, invalid = False, False, False, False
        chars_list = list(w)
        num_valid, num_punc, num_extra, num_english, num_invalid = 0,0,0,0,0
        for d in chars_list:
            if d in allowed_chars_dict["valid"]:
                num_valid += 1
            elif d in allowed_chars_dict["punc"]:
                num_punc += 1
                punc = True
            elif d in allowed_chars_dict["extras"]:
                num_extra += 1
                extra = True
            elif d in allowed_chars_dict["english"]:
                num_english += 1
                english = True
            else:
                num_invalid += 1
                invalid = True

        valid = (not punc) and (not extra) and (not english) and (not invalid)
        
        return [(valid, punc, extra, english, invalid), (num_valid, num_punc, num_extra, num_english, num_invalid)]

    # all_stats = Parallel(n_jobs=32)(delayed(stats_word)(w) for w in modified_words_list)
    all_stats = [stats_word(w) for w in modified_words_list]

    extra_word_list = []
    num_valid_chars, num_valid_words, num_extra_chars, num_extra_words, num_punc_chars, num_punc_words, num_english_chars, num_english_words, num_invalid_chars, num_invalid_words = 0,0,0,0,0,0,0,0,0,0
    for i, ((valid, punc, extra, english, invalid), (num_valid, num_punc, num_extra, num_english, num_invalid)) in enumerate(all_stats):
        num_valid_chars += num_valid
        num_valid_words += valid 
        num_extra_chars += num_extra
        num_extra_words += extra
        num_punc_chars += num_punc
        num_punc_words += punc
        num_english_chars += num_english
        num_english_words += english
        num_invalid_chars += num_invalid
        num_invalid_words += invalid 
        extra_word_list.append(modified_words_list[i])


    if ssplit:
        stats_dict = {
            "clean": norm_out,
            "sents": sentences,
            "num_sents": len(sentences),
            "num_chars": len(norm_out),
            "num_words": len(modified_words_list),
            "num_valid_chars": num_valid_chars,
            "num_valid_words": num_valid_words,
            "num_extra_chars": num_extra_chars,
            "num_extra_words": num_extra_words,
            "num_punc_chars": num_punc_chars,
            "num_punc_words": num_punc_words,
            "num_english_chars": num_english_chars,
            "num_english_words": num_english_words,
            "num_invalid_chars": num_invalid_chars,
            "num_invalid_words": num_invalid_words,
            "extra_words_list": extra_word_list,
            "count_num2words": count_num2words,
        }
    else:
        stats_dict = {
            "clean": norm_out,
            "num_chars": len(norm_out),
            "num_words": len(modified_words_list),
            "num_valid_chars": num_valid_chars,
            "num_valid_words": num_valid_words,
            "num_extra_chars": num_extra_chars,
            "num_extra_words": num_extra_words,
            "num_punc_chars": num_punc_chars,
            "num_punc_words": num_punc_words,
            "num_english_chars": num_english_chars,
            "num_english_words": num_english_words,
            "num_invalid_chars": num_invalid_chars,
            "num_invalid_words": num_invalid_words,
            "extra_words_list": extra_word_list,
            "count_num2words": count_num2words,
        }
        
    return stats_dict

# %%
## NOTE: TESTING SCRIPT

# import regex
# s = "string. With. Some・Really $123 Weird、Non？ASCII。 「（Punctuation）」?र होगा।वहीं, सितसिपास , धीरे-धीरे व"
# remove = regex.compile(r'[\p{C}|\p{P}|\p{S}|\p{Z}]+', regex.UNICODE)
# print(remove.sub(" ", s).strip())

# allowed = allowed_chars(dict_path, lang=LANG)
# stats = clean_and_stats("एक क[1]दम दूर रह गए हैं। इस(this) जीत के बाद जोकोविच एटीपी[sadf(abhigyan) asdf] टेनिस रैंकिंग में नंबर एक खिलाड़ी बने रहेंगे। वहीं, सितसिपास की करियर रैंकिंग में भी सुधार होगा। वे 5वें नंबर से करियर बेस्ट नंबर चार पर पहुंच जाएंगे। दरअसल, तीन साल पहले मॉरिशा की ये बीमारी एक छोटे से पिंपल से शुरु हुई थी। मगर, धीरे-धीरे वह ठीक होने की बजाए फैलता गया।", lang=LANG, allowed_chars_dict=allowed)
# print(stats)

# %%
def get_corpus_stats(corpus_path, dict_path, lang, remove_bracketed_content=True, remove_punctuation=True, normalize=True):

    # Run parallely for all paragraphs
    allowed = allowed_chars(dict_path, lang=lang)
    set_digits = set(DIGITS_SCRIPT[LANG2SCRIPT[lang]] + list(string.digits))

    print('\tLoading Data...')
    corpus = open(corpus_path).readlines()
    print('\tDone Loading Data!')
    print('\tStarting Processing...')
    data_and_stats = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(clean_and_stats)\
        (p, lang, allowed, set_digits, remove_bracketed_content=remove_bracketed_content, \
            remove_punctuation=remove_punctuation, normalize=normalize) for p in tqdm(corpus))
    print('\tDone Processing!')
    return allowed, data_and_stats

# %% [markdown]
# ### Clean Indic-Corp

# %%
def filter_and_clean(dataframe, valid_chars, strict=True, drop_rows=True):
    df_stats = dataframe.describe()
    cutoff_chars_lower = dataframe.num_chars.quantile(0.01) if strict else 2
    cutoff_chars_upper = dataframe.num_chars.quantile(0.95) if strict else dataframe.num_chars.quantile(0.99)
    cutoff_words_lower = dataframe.num_words.quantile(0.01) if strict else 2
    cutoff_words_upper = dataframe.num_words.quantile(0.95) if strict else dataframe.num_chars.quantile(0.99)
    cutoff_extra_chars = 0 if strict else df_stats["num_extra_chars"]["25%"]
    cutoff_english_chars = 0 if strict else df_stats["num_english_chars"]["25%"]

    cutoff_extra_words = 0 if strict else df_stats["num_extra_words"]["25%"] 
    cutoff_english_words = 0 if strict else df_stats["num_english_words"]["25%"]

    # filter and dropna
    df = dataframe.query(f'num_invalid_chars == 0 and num_extra_chars  <= {cutoff_extra_chars} and \
        num_english_chars <= {cutoff_english_chars} and num_extra_words  <= {cutoff_extra_words} and \
        num_english_words <= {cutoff_english_words} and num_chars >= {cutoff_chars_lower} and \
        num_chars <= {cutoff_chars_upper} and num_words >= {cutoff_words_lower} and num_words <= {cutoff_words_upper}').dropna() if drop_rows else dataframe

    valid_str = "".join(list(valid_chars))
    valid_chrs = r'[^'+valid_str+']'
    valid_regex = re.compile(valid_chrs)

    try:
        from pandarallel import pandarallel
        pandarallel.initialize()
        #TODO: verify if the logic to remove multiple spaces in between words is correct
        filtered_sents = df["clean"].parallel_apply(lambda x: " ".join(re.sub(valid_regex, "", str(x)).split()))
    except:
        filtered_sents = df["clean"].apply(lambda x: " ".join(re.sub(valid_regex, "", str(x)).split()))
    
    return filtered_sents, df_stats

# %%
## NOTE: RUNNING FINAL SCRIPT

save = True
if LANG == '':
    parser = argparse.ArgumentParser(description='Cleaning Paras and getting stats')
    parser.add_argument('LangCode',
                        metavar='lang_code',
                        type=str,
                        help='Language Code')
    parser.add_argument('-m', '--manifest', type=str, required=True, help='Manifest_type_prefix, [dc,existing,noa]')
    parser.add_argument('-t', '--type', type=str, required=True, help='Data Type, [train,valid,test,test_known]')
    args = parser.parse_args()  
    lang_code = args.LangCode
    mani_code = args.manifest
    data_type = args.type
    type_name = f'manifest-{mani_code}' if mani_code != '' else 'corpus'
    lang = CODE2LANG[lang_code]
    manifest_dir = indiccorp_path.replace('indic-corp-jun-2022', f'{mani_code}_manifests')
    manifest_path = f'{manifest_dir}/{lang}_{data_type}.wrd'
    if not os.path.exists(manifest_path):
        raise Exception(f'Manifest file NOT FOUND at "{manifest_path}"')

    os.makedirs(f'{dir_path}/{lang}', exist_ok=True)
    dict_path = f'{superb_dict_path}/{lang}_dict.txt'
    # corpus_path = f'{indiccorp_path}/{lang_code}.txt.shuf'
    corpus_path = f'{indiccorp_path}/{lang_code}_sents.txt'

    df_corpus_path = f'{dir_path}/{lang}/{type_name}_sents_and_stats.tsv'
    clean_corpus_path = f'{dir_path}/{lang}/{type_name}_clean_sents.txt'
    extra_words_path = f'{dir_path}/{lang}/{type_name}_extra_words_counter.tsv'
    plots_dir = f'{dir_path}/{lang}/plots'
    os.makedirs(plots_dir, exist_ok=True)

    if mani_code == '':
        print("Cleaning and Stats Calculation #1 ...")
        allowed, data_and_stats = get_corpus_stats(corpus_path, dict_path, lang)
        df = pd.DataFrame(data_and_stats)

        print("\nCleaning, Filtering and Stats Calculation #2 ...")
        clean_data, df_stats = filter_and_clean(dataframe=df, valid_chars = allowed["valid"])
    else:
        print("Cleaning and Stats Calculation #1 ...")
        allowed, data_and_stats = get_corpus_stats(manifest_path, dict_path, lang, remove_bracketed_content=False, remove_punctuation=False, normalize=False)
        df = pd.DataFrame(data_and_stats)

        print("\nCleaning, Filtering and Stats Calculation #2 ...")
        clean_data, df_stats = filter_and_clean(dataframe=df, valid_chars = allowed["valid"], drop_rows=False)

    if save:
        # print('\tSaving Stats and Stats Plots (1/4)')
        # # Set the figure size
        # plt.rcParams["figure.figsize"] = [12, 5]
        # plt.rcParams["figure.autolayout"] = True

        # # Create box-plot for num_valid_chars, num_extra_chars, num_punc_chars, num_english_chars, num_invalid_chars    
        # df[["num_valid_words", "num_extra_words", "num_punc_words", "num_english_words", "num_invalid_words", "count_num2words"]
        # ].plot(kind='box', title=f'[{lang}] Word Level', subplots=True)[0].get_figure().savefig(f'{plots_dir}/{type_name}_word.jpg')

        # plt.rcParams["figure.figsize"] = [10, 5]
        # plt.rcParams["figure.autolayout"] = True
        # # Create box-plot for num_valid_words, num_extra_words, num_punc_words, num_english_words, num_invalid_words
        # df[["num_valid_chars", "num_extra_chars", "num_punc_chars", "num_english_chars", "num_invalid_chars"]
        # ].plot(kind='box', title=f'[{lang}] Character Level', subplots=True)[0].get_figure().savefig(f'{plots_dir}/{type_name}_char.jpg')
        # df_stats.to_csv(f'{plots_dir}/{type_name}_stats.tsv', sep='\t')
        
        # print('\tSaving Final Sentences in txt format (2/4)')
        # clean_data.to_csv(clean_corpus_path,index=False,header=False)

        # print('\tSaving Extra Words and their frequencies (3/4)')
        # extra_words_list = list(itertools.chain.from_iterable(df["extra_words_list"].tolist()))
        # counter = Counter(extra_words_list)
        # top_counter = counter.most_common()
        # df_counter = pd.DataFrame(top_counter, columns=['word', 'freq'])
        # df_counter.to_csv(extra_words_path, sep='\t')

        # print('\tSaving Cleaned Sentences and its complete stats (4/4)')
        # df.to_csv(df_corpus_path, sep='\t')

        # ======================================== new_manifests_path ========================================
        new_manifest_path = manifest_path.replace('manifests/', 'manifests_new/')
        new_manifest_dir = "/".join(new_manifest_path.split('/')[:-1])
        lang_folder_name = new_manifest_path.split('/')[-1].split('_')[0]
        os.makedirs(new_manifest_dir, exist_ok=True)
        lang_dir = new_manifest_dir + '/' + lang_folder_name
        os.makedirs(lang_dir, exist_ok=True)
        clean_data.to_csv(f'{lang_dir}/{data_type}.wrd',index=False,header=False)
        clean_ltrs = clean_data.apply(lambda x: " ".join(list(x.strip().replace(" ", "|"))) + " |")
        clean_ltrs.to_csv(f'{lang_dir}/{data_type}.ltr',index=False,header=False)
        shutil.copyfile(manifest_path.replace('.wrd','.tsv'), f'{lang_dir}/{data_type}.tsv')

    # TODO: Do set operation to filter out repeating sentences. Later take set difference with the test set (before training lm)!

# %% [markdown]
# ### Create lexicon for Indic-Corp

# %%
def create_lexicon(cleaned_sents_df, topk=None, topk_percent=None):
    if (topk is None and topk_percent is None) or (topk is not None and topk_percent is not None):
        raise Exception("Provide exactly one of the parameters 'topk_percent' or 'topk'!")

    print('\tSplitting sentences into word list...')
    try:
        from pandarallel import pandarallel
        pandarallel.initialize()
        splitted_sents = cleaned_sents_df[0].parallel_apply(lambda x: str(x).split())
    except:
        splitted_sents = cleaned_sents_df[0].apply(lambda x: str(x).split())
    
    print('\tCounting word frequencies...')
    words_list = list(itertools.chain.from_iterable(splitted_sents.tolist()))
    counter = Counter(words_list)
    top_counter = counter.most_common()
    df_counter = pd.DataFrame(top_counter, columns=['word', 'freq'])
    try:
        from pandarallel import pandarallel
        pandarallel.initialize()
        df_counter["lexicon"] = df_counter["word"].parallel_apply(lambda x: " ".join(list(x.strip())) + " |")
    except:
        df_counter["lexicon"] = df_counter["word"].apply(lambda x: " ".join(list(x.strip())) + " |")

    tot_words = df_counter["freq"].sum()
    if topk is None and topk_percent is not None:
        topk = int(topk_percent / 100 * len(top_counter))
    elif topk is not None and topk_percent is None:
        topk_percent = topk / len(top_counter) * 100
    
    df_lexicon = df_counter[["word", "lexicon"]].iloc[:topk,:]
    print(f"Total Words: {tot_words}\t Total Unique Words: {len(top_counter)}\t Top-k:{topk}\t %age top-k: {topk_percent:.2f}%")

    return df_counter, df_lexicon

