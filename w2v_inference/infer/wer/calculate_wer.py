import os
import pandas as pd
import numpy as np
import glob
import Levenshtein as Lev
from tqdm import tqdm
import swifter
import argparse
from components import compute_wer

from tqdm import tqdm
from joblib import Parallel, delayed
from indicnlp.tokenize import indic_tokenize, indic_detokenize
from indicnlp.normalize import indic_normalize
from indicnlp.transliterate import unicode_transliterate


def wer( s1, s2):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """

        # build mapping of words to integers
        b = set(s1.split() + s2.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in s1.split()]
        w2 = [chr(word2char[w]) for w in s2.split()]

        return Lev.distance(''.join(w1), ''.join(w2))

def cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2)

def clean_text(row):
    return row[0][0:row.ind]

def preprocess(original_csv):
    original_csv['ind'] = original_csv['text'].str.index('(None')
    original_csv['cleaned_text'] = original_csv.swifter.apply(clean_text, axis = 1)
    return original_csv

def calculate_wer(row):
    wer_local = ''
    try:
        wer_local = wer(row['original'], row['predicted'])
        #cer_local = cer(row['cleaned_text'], row['text_y'])
    except:
        print(row)
        return len(row['original'].split(' '))
    return wer_local


def calculate_cer(row):
    try:
        cer_local = cer(row['original'], row['predicted'])
    except:
        return len(row['original'].str.replace(' ','').str.len())
    return cer_local
# /home/abhigyan/gcp/alignment/w2v-inference/results/telugu_mucs_large_comb_128/sentence_wise_wer.csv

def preprocess_line(line, normalizer, src, tgt):
    if tgt == "hi":
        return unicode_transliterate.UnicodeIndicTransliterator.transliterate(
            " ".join(
                indic_tokenize.trivial_tokenize(normalizer.normalize(line.strip()), src)
            ),
            src,
            tgt,
        )
    else:
        wrds = indic_tokenize.trivial_tokenize(line.strip())
        w_updated = []
        for w in wrds:
            lst = list(w)
            updated = []
            next_ = True
            for l in range(len(lst)):
                if l != len(lst) - 1:
                    if hex(ord(lst[l])) == "0x930" and hex(ord(lst[l + 1])) == "0x93c":
                        updated.append(chr(0x931))
                        next_ = False
                        continue
                    elif (
                        hex(ord(lst[l])) == "0x933" and hex(ord(lst[l + 1])) == "0x93c"
                    ):
                        updated.append(chr(0x934))
                        next_ = False
                        continue
                    elif (
                        hex(ord(lst[l])) == "0x928" and hex(ord(lst[l + 1])) == "0x93c"
                    ):
                        updated.append(chr(0x929))
                        next_ = False
                        continue
                if next_:
                    updated.append(lst[l])
                else:
                    next_ = True
            w_updated.append("".join(updated))

        line = " ".join(w_updated)

        return indic_detokenize.trivial_detokenize(
            unicode_transliterate.UnicodeIndicTransliterator.transliterate(
                line.strip(),
                src,
                tgt,
            )
        )


def preprocess(infname, outfname, src, tgt):
    """
    Normalize the data, tokenize the data and convert to devanagari
    """
    num_lines = sum(1 for _ in open(infname, "r"))
    normfactory = indic_normalize.IndicNormalizerFactory()
    normalizer = normfactory.get_normalizer(src)

    with open(infname, "r", encoding="utf-8") as infile, open(
        outfname, "w", encoding="utf-8"
    ) as outfile:
        outlines = Parallel(n_jobs=-1, backend="multiprocessing")(
            delayed(preprocess_line)(line, normalizer, src, tgt)
            for line in tqdm(infile, total=num_lines)
        )
        for line in outlines:
            outfile.write(line + "\n")
    return True

def run_pipeline(ground_truth, predicted, args_local=None):
    print(ground_truth,predicted)
    with open(ground_truth, encoding='utf-8') as file:
        original_csv = file.readlines()
    original_csv = [line.strip() for line in original_csv]

    with open(predicted, encoding='utf-8') as file:
        raw_lines = file.read().strip().split("\n")
    line2sen = {}
    for line in raw_lines:
        sen, lno  = line.split("(None-")
        lno = lno.split(')')[0]
        line2sen[int(lno)] = sen.strip()
    fol_name=(args_local.name).replace("/sentence_wise_wer.csv","")
    # print(fol_name)
    # fol_name="/".join((args_local.name).split('/')[:-1])
    with open(fol_name+"/temp.wrd",'w') as f:
        for n in range(len(line2sen)):
            print(line2sen[n],file=f)

    if args_local.transl:
        lang_map = {'hindi': 'hi', "gujarati":'gu', 'odia':'or', 'marathi':'hi',
                    'tamil':'ta', 'telugu':'te'}
        target = lang_map[args_local.lang]
        preprocess(fol_name+"/temp.wrd", fol_name+"/tr_temp.wrd", 'hi', target)
    
        with open(fol_name+"/tr_temp.wrd", encoding='utf-8') as file:
            predicted_csv = file.readlines()
        predicted_csv = [line.strip() for line in predicted_csv]

    else:
        with open(fol_name+"/temp.wrd", encoding='utf-8') as file:
            predicted_csv = file.readlines()
        predicted_csv = [line.strip() for line in predicted_csv]

    print(len(original_csv)," ", len(predicted_csv))

    data = list(zip(original_csv, predicted_csv))
    df_merged = pd.DataFrame(data, columns = ['original', 'predicted'])
    
    df_merged['wer'] = df_merged.apply(calculate_wer, axis = 1)
    df_merged['cer'] = df_merged.swifter.apply(calculate_cer, axis = 1)
    df_merged['num_tokens'] = df_merged['original'].str.split().str.len()
    df_merged['num_chars'] = df_merged['original'].str.replace(' ','').str.len()
    
    df_merged.sort_values(by = 'wer', ascending=False)
    fwer = df_merged.wer.sum() / df_merged.num_tokens.sum()
    fcer = df_merged.cer.sum() / df_merged.num_chars.sum()
    print('WER: ', fwer*100)
    print('CER: ', fcer*100)
    wer = round(fwer*100,2)
    cer = round(fcer*100,2)
    df_merged.rename(columns={'wer': f'wer:{round(fwer*100,2)}', 'cer': f'cer:{round(fcer*100,2)}'}, inplace=True)
    return df_merged,wer,cer
    

def calculate_errors(row):
    ret_object = compute_wer(predictions=[row['predicted']], references=[row['original']])
    return [ret_object['substitutions'], ret_object['insertions'], ret_object['deletions']]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process CER pipeline')
    parser.add_argument('-o', '--original', required=True, help='Original File')
    parser.add_argument('-p', '--predicted', required=True, help='Predicted File')
    parser.add_argument('-s', '--save-output', help='save output file', type=bool)
    parser.add_argument('-n', '--name', help='save output file name', type=str)
    parser.add_argument('-t','--tsv', type=str)
    parser.add_argument('-e', '--sid', type=bool)
    parser.add_argument('--transl', type=bool, default=False)
    parser.add_argument('--lang', type=str) #hindi, gujarati, odia, marathi, tamil, telugu

    args_local = parser.parse_args()
    df,wer,cer = run_pipeline(args_local.original, args_local.predicted, args_local)

    if args_local.sid:
        ret_object= df.swifter.apply(calculate_errors, axis=1)
        df['errors'] = ret_object
        df_errors = pd.DataFrame(df['errors'].to_list(), columns=['substitutions','insertions', 'deletions'])
        df = pd.concat([df, df_errors], axis=1)
        df = df.drop(columns=['errors'])

    if args_local.save_output:
        df.to_csv(args_local.name, index=False)
        with open('~/WER_valid.csv', 'a+') as f:
            name=args_local.name.split('/')[-2]
            f.write(f'{name},{wer},{cer}\n')

