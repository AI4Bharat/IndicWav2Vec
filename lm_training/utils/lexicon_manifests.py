LANG = ''

from collections import Counter
import os
import string
from tqdm import tqdm
import pandas as pd
import re
import itertools
import time
import argparse

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

def create_lexicon(cleaned_sents_df, topk=None, topk_percent=None):
    if (topk is None and topk_percent is None) or (topk is not None and topk_percent is not None):
        raise Exception("Provide exactly one of the parameters 'topk_percent' or 'topk'!")

    print('\tSplitting sentences into word list...')
    try:
        import pgmpy
        from pandarallel import pandarallel
        pandarallel.initialize()
        splitted_sents = cleaned_sents_df[0].parallel_apply(lambda x: str(x).split())
    except:
        splitted_sents = cleaned_sents_df[0].apply(lambda x: str(x).split())
    
    del cleaned_sents_df
    print('\tCounting word frequencies...')
    words_list = list(itertools.chain.from_iterable(splitted_sents.tolist()))
    counter = Counter(words_list)
    top_counter = counter.most_common()
    df_counter = pd.DataFrame(top_counter, columns=['word', 'freq'])
    try:
        import pgmpy
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

save = True
if LANG == '':
    parser = argparse.ArgumentParser(description='Obtaining Lexicon')
    parser.add_argument('LangCode',
                        metavar='lang_code',
                        type=str,
                        help='Language Code')
    parser.add_argument('-m', '--manifest', type=str, default='', help='Manifest_type_prefix, [dc,existing,noa]')
    args = parser.parse_args()  
    lang_code = args.LangCode
    mani_code = args.manifest
    type_name = f'manifest-{mani_code}' if mani_code != '' else 'corpus'
    lang = CODE2LANG[lang_code]

    manifest_path = f'{dir_path}/{lang}/{type_name}_clean_sents.txt'
    if not os.path.exists(manifest_path):
        raise Exception(f'Manifest file NOT FOUND at "{manifest_path}"')

    lexicon_path = f'{dir_path}/{lang}/{type_name}_lexicon.txt'
    counter_path = f'{dir_path}/{lang}/{type_name}_words_counter.tsv'

    print("Loading Data ...")
    df = pd.read_csv(manifest_path, header=None, sep='\t')

    print("Creating Lexicon ...")
    df_counter, df_lexicon = create_lexicon(cleaned_sents_df=df, topk_percent=100)
    del df
    if save:
        print('Saving Lexicon and Word Counter files...')
        df_counter.to_csv(counter_path, sep='\t')
        df_lexicon.to_csv(lexicon_path, sep='\t', header=False, index=False)
        # Set the figure size

