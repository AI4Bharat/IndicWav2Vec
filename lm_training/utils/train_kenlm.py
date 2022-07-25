#This code has been taken in part from https://github.com/mozilla/DeepSpeech/blob/master/data/lm/generate_lm.py

import argparse
import os
import subprocess
from tqdm import tqdm
import shutil
import pandas as pd
import glob

CODE2LANG = {"bn": "bengali", "gu": "gujarati", "hi": "hindi", "kn": "kannada", "ml": "malayalam",\
     "mr": "marathi", "or": "odia", "pa": "punjabi", "sa": "sanskrit", "ta": "tamil", "te": "telugu", "ur": "urdu" } 

LANG2CODE = dict((v, k) for k, v in CODE2LANG.items())

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def merge_text_files(args, lang):
    os.system(f"cat {args.lm_base_dirpath}/{lang}/C_*_sents.txt {args.lm_base_dirpath}/{lang}/M_*_sents.txt > {args.lm_base_dirpath}/{lang}/ALL_SENTS.txt")

def merge_word_counters_corpus(args, lang):
    corpus_counter_paths = glob.glob(f'{args.lm_base_dirpath}/{lang}/C_*_counter.tsv')
    df_corpus = None
    for c_path in corpus_counter_paths:
        df_temp = pd.read_csv(c_path, sep='\t')
        if df_corpus is not None:
            df_temp.rename(columns = {'freq':'freq_new'}, inplace = True)
            df_corpus = pd.merge(df_corpus, df_temp, on='word', how='left')
            df_corpus['freq'] = df_corpus['freq'] + df_corpus['freq_new'].fillna(0)
            df_corpus.drop('freq_new', axis=1, inplace=True)
        else:
            df_corpus = df_temp
    if df_corpus is not None:
        df_corpus_sort = df_corpus[['word','freq']].sort_values(by='freq', ascending=False)
        df_corpus_sort.to_csv(f'{args.lm_base_dirpath}/{lang}/C_ALL_COUNTER.tsv', sep='\t')
    
def merge_word_counters_manifests(args, lang):
    corpus_counter_paths = glob.glob(f'{args.lm_base_dirpath}/{lang}/M_*_counter.tsv')
    df_manifests = None  
    for c_path in corpus_counter_paths:
        df_temp = pd.read_csv(c_path, sep='\t')
        if df_manifests is not None:
            df_temp.rename(columns = {'freq':'freq_new'}, inplace = True)
            df_manifests = pd.merge(df_manifests, df_temp, on='word', how='left')
            df_manifests['freq'] = df_manifests['freq'] + df_manifests['freq_new'].fillna(0)
            df_manifests.drop('freq_new', axis=1, inplace=True)
        else:
            df_manifests = df_temp
    if df_manifests is not None:
        df_manifests_sort = df_manifests[['word','freq']].sort_values(by='freq', ascending=False)
        df_manifests_sort.to_csv(f'{args.lm_base_dirpath}/{lang}/M_ALL_COUNTER.tsv', sep='\t')

def prepare_and_filter_topk(args, lang):
    topk = args.topk
    topk_percent = args.topk_percent
    min_freq = args.min_freq
    if topk is None and topk_percent is None and min_freq is None:
        raise Exception("Provide exactly one of the parameters 'topk_percent' or 'topk'!")

    print("\nStep 1:\tPreparing Data for LM-------------------------------------------")
    c_all_counter_path = f'{args.lm_base_dirpath}/{lang}/C_ALL_COUNTER.tsv'
    m_all_counter_path = f'{args.lm_base_dirpath}/{lang}/M_ALL_COUNTER.tsv'
    all_sents_path = f'{args.lm_base_dirpath}/{lang}/ALL_SENTS.txt'

    if os.path.exists(all_sents_path):
        print('WARNING: "ALL_SENTS.txt" already exists, skipping...')
    else:
        print('Merging all "*clean_sents.txt" files!...')
        merge_text_files(args, lang)
        
    temp_sents_path = f'{args.lm_base_dirpath}/{lang}/{args.intermediate_dir}/temp_sents.txt.gz'
    if os.path.exists(temp_sents_path):
        print('WARNING: "temp_sents.txt.gz" already exists, skipping...')
    else:
        print("Creating a temporary copy of ALL_SENTS!...")
        _ = shutil.copyfile(all_sents_path, temp_sents_path)
    if not args.do_filtering:
        return temp_sents_path, None, None

    if os.path.exists(c_all_counter_path):
        print('WARNING: "C_ALL_COUNTER.tsv" already exists, skipping...')
    else:
        print('Merging all "C_*words_counters.tsv" files!...')
        merge_word_counters_corpus(args, lang)
    if os.path.exists(m_all_counter_path):
        print('WARNING: "M_ALL_COUNTER.tsv" already exists, skipping...')
    else:
        print('Merging all "M_*words_counters.tsv" files!...')
        merge_word_counters_manifests(args, lang)
        
    if os.path.exists(c_all_counter_path):
        df_corpus_counter = pd.read_csv(c_all_counter_path, sep='\t')
        tot_words = df_corpus_counter["freq"].sum()
        if topk is not None:
            topk_percent = topk / len(df_corpus_counter) * 100
            df_corpus_topk = df_corpus_counter.iloc[:topk,:]
        elif topk_percent is not None:
            topk = int(topk_percent / 100 * len(df_corpus_counter))
            df_corpus_topk = df_corpus_counter.iloc[:topk,:]
        elif min_freq is not None:
            df_corpus_topk = df_corpus_counter[df_corpus_counter["freq"]>=min_freq]
            topk = len(df_corpus_topk)
            topk_percent = topk / len(df_corpus_counter) * 100

        print(f"Corpus Stats: Total Words: {tot_words}\t Total Unique Words: {len(df_corpus_counter)}\t Top-k:{topk}\t %age top-k: {topk_percent:.2f}%")
        if os.path.exists(m_all_counter_path):
            df_manifests_counter = pd.read_csv(m_all_counter_path, sep='\t')
            df_all_counter = df_corpus_topk[["word","freq"]].merge(df_manifests_counter[["word","freq"]], on='word', how='left')
        else:
            df_all_counter = df_corpus_topk[["word","freq"]]
            
    elif os.path.exists(m_all_counter_path):
        df_manifests_counter = pd.read_csv(m_all_counter_path, sep='\t')
        df_all_counter = df_manifests_counter[["word","freq"]]
        
    else:
        print("ERROR: Unable to create lexicon file as no word counter files was found!")
        return temp_sents_path, None, None

    df_all_counter["lexicon"] = df_all_counter["word"].apply(lambda x: " ".join(list(x.strip())) + " |")
    lexicon_path = f'{args.lm_base_dirpath}/{lang}/{args.lm_dirname}/topk-{topk}_lexicon.lst'
    
    c_m_topk_counter_path = f'{args.lm_base_dirpath}/{lang}/C_M_{topk}_COUNTER.tsv'
    df_all_counter[["word","lexicon"]].to_csv(c_m_topk_counter_path, sep="\t")
    df_all_counter[["word","lexicon"]].to_csv(lexicon_path, sep="\t", index=False, header=False)

    print(f"Combined Unique Words in the Lexicon, #{len(df_all_counter)}")

    return temp_sents_path, lexicon_path, topk


def build_lm(args, lang, sents_path, lexicon_path=None, topk=None):
    
    print("\nStep 2:\tBuilding LM-------------------------------------------")
    output_dir = f'{args.lm_base_dirpath}/{lang}/{args.lm_dirname}'
    intermediate_dir = f'{args.lm_base_dirpath}/{lang}/{args.intermediate_dir}'
    lm_path = os.path.join(intermediate_dir, "lm.arpa")

    if not os.path.exists(lm_path):
        print("Creating ARPA file ...\n")
        subargs = [
                os.path.join(args.kenlm_bins, "lmplz"),
                "--order",
                str(args.arpa_order),
                "--temp_prefix",
                intermediate_dir,
                "--memory",
                args.max_arpa_memory,
                "--text",
                sents_path,
                "--arpa",
                lm_path,
                "--prune",
                *args.arpa_prune.split("|"),
            ]
        if args.discount_fallback:
            subargs += ["--discount_fallback"]
        subprocess.check_call(subargs)

    # Filter LM using vocabulary of top-k words
    filtered_path = os.path.join(intermediate_dir, "lm_filtered.arpa")
    if not os.path.exists(filtered_path) and args.do_filtering and lexicon_path is not None:
        print("\nFiltering ARPA file using vocabulary of top-k words ...")
        lexicon = open(lexicon_path).read()
        subprocess.run(
            [
                os.path.join(args.kenlm_bins, "filter"),
                "single",
                "model:{}".format(lm_path),
                filtered_path,
            ],
            input=lexicon.encode("utf-8"),
            check=True,
        )
    else:
        filtered_path = lm_path

    # Quantize and produce trie binary.
    print("\nBuilding lm.binary ...")
    if topk is not None:
        binary_path = os.path.join(output_dir, f"topk-{topk}_lm.binary")
    else:
        binary_path = os.path.join(output_dir, f"lm.binary")
    subprocess.check_call(
        [
            os.path.join(args.kenlm_bins, "build_binary"),
            "-a",
            str(args.binary_a_bits),
            "-q",
            str(args.binary_q_bits),
            "-v",
            args.binary_type,
            filtered_path,
            binary_path,
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate lm.binary."
    )
    parser.add_argument(
        'LangCode', metavar='lang_code', type=str, help='Language Code')
    parser.add_argument(
        "--lm_base_dirpath", help="Directory path for the base lm", type=str, required=True
    )
    parser.add_argument(
        "--lm_dirname", help="Directory name for the lm folder", type=str, default="lm"
    )
    parser.add_argument(
        "--intermediate_dir", help='Specify the name of the intermediate directory. Pass "" if not planning to store intermediate results', type=str, default="itms"
    )
    parser.add_argument(
        "--topk",
        help="Use topk most frequent words for the vocab.txt file. These will be used to filter the ARPA file.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--topk_percent",
        help="Use topk most frequent words for the vocab.txt file. These will be used to filter the ARPA file.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--min_freq",
        help="Minimum frequency of words for the vocab.txt file. These will be used to filter the ARPA file.",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--kenlm_bins",
        help="File path to the KENLM binaries lmplz, filter and build_binary",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--arpa_order",
        help="Order of k-grams in ARPA-file generation",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--max_arpa_memory",
        help="Maximum allowed memory usage for ARPA-file generation",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--arpa_prune",
        help="ARPA pruning parameters. Separate values with '|'",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--binary_a_bits",
        help="Build binary quantization value a in bits",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--binary_q_bits",
        help="Build binary quantization value q in bits",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--binary_type",
        help="Build binary data structure type",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--do_filtering",
        help="Perform filtering of kenlm model using lexicon",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--prepare_lm_data_lexicon",
        help="Combine sents data and create lexicon from topk word occurances",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--clean_build",
        help="Delete old intermediate files and build",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--discount_fallback",
        help="To try when such message is returned by kenlm: 'Could not calculate Kneser-Ney discounts [...] rerun with --discount_fallback'",
        action="store_true",
    )

    args = parser.parse_args()

    lang_code = args.LangCode
    lang = CODE2LANG[lang_code]

    output_dir = f'{args.lm_base_dirpath}/{lang}/{args.lm_dirname}'
    os.makedirs(output_dir, exist_ok=True)

    remove_intermediate = False
    if args.intermediate_dir == "":
        args.intermediate_dir = "temp_dir"
        remove_intermediate = True
    intermediate_dir = f'{args.lm_base_dirpath}/{lang}/{args.intermediate_dir}'
    os.makedirs(intermediate_dir, exist_ok=True)

    if args.clean_build:
        # Delete intermediate files
        if os.path.exists(f'{args.lm_base_dirpath}/{lang}/*_ALL_*'):
            os.system(f'rm {args.lm_base_dirpath}/{lang}/*_ALL_*')
        if os.path.exists(output_dir):
            os.system(f'rm {output_dir}/*')
        if os.path.exists(os.path.join(intermediate_dir, "temp_sents.txt.gz")):
            os.remove(os.path.join(intermediate_dir, "temp_sents.txt.gz")) 
        if os.path.exists(os.path.join(intermediate_dir, "lm.arpa")):
            os.remove(os.path.join(intermediate_dir, "lm.arpa")) 
        if os.path.exists(os.path.join(intermediate_dir, "lm_filtered.arpa")):
            os.remove(os.path.join(intermediate_dir, "lm_filtered.arpa"))
        
    if args.prepare_lm_data_lexicon:
        sents_path, lexicon_path, topk = prepare_and_filter_topk(args, lang)
        build_lm(args, lang, sents_path, lexicon_path, topk)
    else:
        if args.sent_path is not None:
            build_lm(args, lang, args.sents_path, args.lexicon_path, args.topk)
        else:
            print('ERROR: LM build failed as "sents_path" was not provided and "prepare_lm_data_lexicon" was None!')

    if remove_intermediate:
        # Delete intermediate files
        if os.path.exists(os.path.join(intermediate_dir, "temp_sents.txt.gz")):
            os.remove(os.path.join(intermediate_dir, "temp_sents.txt.gz")) # Do keep temp_sents.txt.gz
        if os.path.exists(os.path.join(intermediate_dir, "lm.arpa")):
            os.remove(os.path.join(intermediate_dir, "lm.arpa")) # Do keep lm.arpa
    
    # remove combined manifests and text data
    if os.path.exists(f'{args.lm_base_dirpath}/{lang}/*_ALL_*'):
        os.system(f'rm {args.lm_base_dirpath}/{lang}/*_ALL_*')
    # remove lm_filtered.arpa 
    if os.path.exists(os.path.join(intermediate_dir, "lm_filtered.arpa")):
        os.remove(os.path.join(intermediate_dir, "lm_filtered.arpa"))


if __name__ == "__main__":
    main()
