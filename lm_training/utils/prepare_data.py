'''
Generates clean text file and word counter tsv files from corpus/manifest data and provided dictionary for 13 Indian Languages namely, 
hindi, marathi, sanskrit, bengali, punjabi, gujarati, odia, tamil, telugu, kannada, malayalam, sinhala and urdu!
Optionally, also creates new (cleaned) manifest files in exactly the same format!
'''
# Import Statements

from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize.sentence_tokenize import sentence_split
from num_to_words.num_to_words import num_to_word
from joblib import Parallel, delayed
from operator import itemgetter
from matplotlib import pyplot as plt

from collections import Counter
import regex
import os
import string
from tqdm import tqdm
import pandas as pd
import re
import itertools
import time
import shutil 

# Setup argparse
import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Setup logging (TODO)

# Define all GLOBAL Vars
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

def ssplit_paragraphs(input, lang):
    lang_code = LANG2CODE[lang]
    return sentence_split(input, lang_code)

def get_sents(corpus_path, lang):
    print('Loading Data')
    corpus = open(corpus_path).readlines()
    print('Done Loading Data')
    ssplit_list = Parallel(n_jobs=-1)(delayed(ssplit_paragraphs)(p, lang) for p in tqdm(corpus))
    return list(itertools.chain.from_iterable(ssplit_list))

def clean_and_stats(input, lang, allowed_chars_dict, set_digits, ssplit=False, remove_bkts=True, remove_puncts=True, do_normalize=True, do_num2word=True):
    '''
    Stats to return are, length, num_words, allowed_char_stats, other_stats
    '''
    lang_code = LANG2CODE[lang]

    # Remove bracketed content
    no_bracs = re.sub("[\(\[].*?[\)\]]", "", input) if remove_bkts else input
    # no_puncs = re.sub(r'[^\w\s]','',no_bracs)

    # Remove punctuations TODO: If the logic also includes indian puncts
    remove = regex.compile(r'[\p{C}|\p{P}|\p{S}|\p{Z}]+', regex.UNICODE)
    no_puncs = remove.sub(" ", no_bracs).strip() if remove_puncts else no_bracs

    # Normalize and num2word
    norm_input = normalize_sentence(no_puncs, lang_code) if do_normalize else no_puncs
    # norm_input = no_puncs # Only for Urdu
    words_list = norm_input.split()
  
    def get_num2word(w):
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
    if do_num2word and lang_code not in ['ml', 'ur', 'sa']:
        num2word_out_list = [get_num2word(w) for w in words_list]
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

def get_corpus_stats(corpus_path, dict_path, lang, remove_bkts=True, remove_puncts=True, do_normalize=True, do_num2word=False):

    # Run parallely for all paragraphs
    allowed = allowed_chars(dict_path, lang=lang)
    set_digits = set(DIGITS_SCRIPT[LANG2SCRIPT[lang]] + list(string.digits))

    print('\tLoading Data...')
    corpus = open(corpus_path).readlines()
    print('\tDone Loading Data!')
    print('\tStarting Processing...')
    data_and_stats = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(clean_and_stats)\
        (p, lang, allowed, set_digits, remove_bkts=remove_bkts, \
            remove_puncts=remove_puncts, do_normalize=do_normalize) for p in tqdm(corpus))
    print('\tDone Processing!')
    return allowed, data_and_stats

def filter_and_clean(dataframe, valid_chars, drop_rows="off"):
    df_stats = dataframe.describe()
    if drop_rows != "off":
        strict = True if drop_rows == "strict" else False
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
            num_chars <= {cutoff_chars_upper} and num_words >= {cutoff_words_lower} and num_words <= {cutoff_words_upper}').dropna()
    else:
        df = dataframe
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
    
    # words_list = list(itertools.chain.from_iterable(filtered_sents))
    words_list = " ".join(filtered_sents).split()
    counter = Counter(words_list)
    top_counter = counter.most_common()
    df_counter = pd.DataFrame(top_counter, columns=['word', 'freq'])

    return filtered_sents, df_counter, df_stats

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Prepare Data: Clean corpus, create word frequency count!')
    parser.add_argument('LangCode',
                        metavar='lang_code',
                        type=str,
                        help='Language Code')
    parser.add_argument('-d', '--data_dir', type=str, required=True, help='Directory containing corpus/fairseq-style-data-manifests data')
    parser.add_argument('--data_type', type=str, required=True, choices=["C", "M"], help='Data Type: [C, M] (c for Corpus, m for Manifest). Need to specify as they have separate processing pipeline.')
    parser.add_argument('--split_type', type=str, default='train', help='Split Type: [train,valid,test,test_known] NOTE: This is ONLY for fairseq-style-data-manifests folder')
    parser.add_argument('--dict_dir', type=str, default=None, help="Directory containing language dictionary, if None, defaults to dict.ltr.txt")
    parser.add_argument('--out_dir', type=str, default='./Model', help="Default Directory to store the plots/models etc.")
    parser.add_argument('--save_plots', type=str2bool, default=True, help="do `--save_plots=True` for saving plots/stats.")
    parser.add_argument('--save_itms', type=str2bool, default=True, help="do `--save_itms=True` to save intermediary models.")
    parser.add_argument('--save_new_manifests', type=str2bool, default=False, help="It creates new manifest folder. NOTE: This is ONLY for fairseq-style-data-manifests folder")
    parser.add_argument('--new_manifests_dirname', type=str2bool, default=False, help="Path to the new manifests folder. NOTE: This is ONLY for fairseq-style-data-manifests folder")
    parser.add_argument('--do_normalize', type=str2bool, default=True, help="Perform Normalization of Indian Characters")
    parser.add_argument('--do_num2word', type=str2bool, default=True)
    parser.add_argument('--drop_rows', type=str, default="off", choices=["on", "off", "strict"], help="Drop rows which have any foreign chars (after normalizing and pre-processing text!)")
    parser.add_argument('--remove_puncts', type=str2bool, default=True)
    parser.add_argument('--remove_bkts', type=str2bool, default=True, help="If True, removes brackets along with the content")

    args = parser.parse_args()
    lang_code = args.LangCode
    lang = CODE2LANG[lang_code]

    data_dir = args.data_dir
    basename = os.path.basename(data_dir)
    data_type = args.data_type
    split_type = args.split_type
    out_dir = args.out_dir
    out_lang_dir = os.path.join(out_dir, lang)
    do_normalize = args.do_normalize
    do_num2word = args.do_num2word
    remove_puncts = args.remove_puncts
    remove_bkts = args.remove_bkts
    drop_rows = args.drop_rows

    os.makedirs(out_lang_dir, exist_ok=True)

    if args.dict_dir is not None:
        dict_filepath = os.path.join(args.dict_dir, f"{lang}_dict.txt")
    else:
        dict_filepath = os.path.join(data_dir, lang, 'dict.ltr.txt')
    
    if data_type == "M":
        txt_filepath = os.path.join(data_dir, f'{lang}_{split_type}.wrd')
        # txt_filepath = os.path.join(data_dir, lang, f'{split_type}.wrd')
    elif data_type == "C":
        txt_filepath = os.path.join(data_dir, f'{lang_code}_sents.txt')
    
    if not os.path.exists(txt_filepath):
        raise Exception(f'Text file NOT FOUND at "{txt_filepath}"')
    if not os.path.exists(dict_filepath):
        raise Exception(f'Dict file NOT FOUND at "{dict_filepath}"')

    if args.save_itms:
        itms_dir = os.path.join(out_lang_dir, 'itms')
        os.makedirs(itms_dir, exist_ok=True)
    if args.save_plots:
        plots_dir = os.path.join(out_lang_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

    if data_type == 'C':
        print("----------------------------------------------------------------------------")
        print("-------------------Cleaning and Stats Calculation #1 ...-------------------")
        allowed, data_and_stats = get_corpus_stats(txt_filepath, dict_filepath, lang, remove_bkts=remove_bkts, remove_puncts=remove_puncts, do_normalize=do_normalize, do_num2word=do_num2word)
        df = pd.DataFrame(data_and_stats)
        print("\n-------------------Cleaning, Filtering and Stats Calculation #2 ...-------------------")
        clean_data, df_counter, df_stats = filter_and_clean(dataframe=df, valid_chars = allowed["valid"], drop_rows=drop_rows)
        out_fname_prefix = f"{data_type}_{basename}"

    elif data_type == 'M':
        print("-------------------Cleaning and Stats Calculation #1 ...-------------------")
        allowed, data_and_stats = get_corpus_stats(txt_filepath, dict_filepath, lang, remove_bkts=False, remove_puncts=remove_puncts, do_normalize=do_normalize, do_num2word=do_num2word)
        df = pd.DataFrame(data_and_stats)

        print("\n-------------------Cleaning, Filtering and Stats Calculation #2 ...-------------------")
        clean_data, df_counter, df_stats = filter_and_clean(dataframe=df, valid_chars = allowed["valid"], drop_rows=drop_rows)
        out_fname_prefix = f"{data_type}_{basename}_{split_type}"

    print('\tSaving Final Sentences in txt format...')
    clean_data_path = os.path.join(out_lang_dir,f'{out_fname_prefix}_clean_sents.txt')
    clean_data.to_csv(clean_data_path,index=False,header=False)

    print('\tSaving Unique Words and their frequencies...')
    words_counter_path = os.path.join(out_lang_dir,f'{out_fname_prefix}_words_counter.tsv')
    df_counter.to_csv(words_counter_path, sep='\t')

    if args.save_plots:
        print('\tSaving Stats and Stats Plots')
        # Set the figure size
        plt.rcParams["figure.figsize"] = [12, 5]
        plt.rcParams["figure.autolayout"] = True

        # Create box-plot for num_valid_chars, num_extra_chars, num_punc_chars, num_english_chars, num_invalid_chars    
        df[["num_valid_words", "num_extra_words", "num_punc_words", "num_english_words", "num_invalid_words", "count_num2words"]
        ].plot(kind='box', title=f'[{lang}] Word Level', subplots=True)[0].get_figure().savefig(f'{plots_dir}/{out_fname_prefix}_word.jpg')

        plt.rcParams["figure.figsize"] = [10, 5]
        plt.rcParams["figure.autolayout"] = True
        # Create box-plot for num_valid_words, num_extra_words, num_punc_words, num_english_words, num_invalid_words
        df[["num_valid_chars", "num_extra_chars", "num_punc_chars", "num_english_chars", "num_invalid_chars"]
        ].plot(kind='box', title=f'[{lang}] Character Level', subplots=True)[0].get_figure().savefig(f'{plots_dir}/{out_fname_prefix}_char.jpg')
        df_stats.to_csv(f'{plots_dir}/{out_fname_prefix}_stats.tsv', sep='\t')

    if args.save_itms:
        clean_sents_stats_path = os.path.join(itms_dir, f'{out_fname_prefix}_clean_sents_stats.tsv')
        extra_words_counter_path = os.path.join(itms_dir, f'{out_fname_prefix}_extra_words_counter.tsv')
        print('\tSaving Extra Words and their frequencies (1/2)')
        extra_words_list = list(itertools.chain.from_iterable(df["extra_words_list"].tolist()))
        counter = Counter(extra_words_list)
        top_counter = counter.most_common()
        df_counter = pd.DataFrame(top_counter, columns=['word', 'freq'])
        df_counter.to_csv(extra_words_counter_path, sep='\t')

        print('\tSaving Cleaned Sentences and its complete stats (2/2)')
        df.to_csv(clean_sents_stats_path, sep='\t')

    # TODO: BUGS...Check for file paths (correct it using basename, os.path.join etc.)
    if data_type=="M" and args.save_new_manifests:
        # ======================================== new_manifests_path ========================================
        if args.new_manifests_dirname is None:
            new_manifest_path = txt_filepath.replace('_manifests/', '_manifests_new/')
            new_manifest_dir = "/".join(new_manifest_path.split('/')[:-1])
        else:
            new_manifest_dir = args.new_manifests_dirname
            new_manifest_path = os.path.join(new_manifest_dir, f'{lang}_{split_type}.wrd')
            # new_manifest_path = os.path.join(new_manifest_dir, lang, f'{split_type}.wrd')
            lang_folder_name = lang

        # lang_folder_name = new_manifest_path.split('/')[-1].split('_')[0]
        os.makedirs(new_manifest_dir, exist_ok=True)
        lang_dir = new_manifest_dir + '/' + lang_folder_name
        os.makedirs(lang_dir, exist_ok=True)
        clean_data.to_csv(f'{lang_dir}/{split_type}.wrd',index=False,header=False)
        clean_ltrs = clean_data.apply(lambda x: " ".join(list(x.strip().replace(" ", "|"))) + " |")
        clean_ltrs.to_csv(f'{lang_dir}/{split_type}.ltr',index=False,header=False)
        shutil.copyfile(txt_filepath.replace('.wrd','.tsv'), f'{lang_dir}/{split_type}.tsv')
        shutil.copyfile(dict_filepath, f'{lang_dir}/dict.ltr.txt')

    # TODO: Do set operation to filter out repeating sentences. Later take set difference with the test set (before training lm)!
    print("----------------------------------------------------------------------------")
