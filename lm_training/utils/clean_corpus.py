import pandas as pd
from tqdm import tqdm
import string
import re
import os
import glob
import itertools
from collections import Counter
import argparse
import re

def main(args):
    lang = args.lang
    dir_path = args.dir_path
    os.makedirs(f'{dir_path}/{lang}', exist_ok=True)
    dict_path = f'{dir_path}/{lang}/dict.txt'
    clean_dump = f'{dir_path}/{lang}/clean_dump.txt'
    lexicon_kenlm = f'{dir_path}/{lang}/lexicon.txt'
    clean_toks = f'{dir_path}/{lang}/clean_toks.txt'

    dict_range_st = args.st # 
    dict_range_en = args.en # ref: https://www.ssec.wisc.edu/~tomw/java/unicode.html

    gen_punc_st = 8192
    gen_punc_en = 8334 # ref: https://www.ssec.wisc.edu/~tomw/java/unicode.html

    dict_df = pd.read_csv(dict_path, sep=' ', header= None, names=['char', 'ul'])
    dict_chars = set(dict_df['char'].tolist())

    all_chars = set([chr(i) for i in range(dict_range_st, dict_range_en+1)])
    print(f'Extra characters: {len(dict_chars-all_chars)}\n {dict_chars-all_chars}')
    print(f'Missing characters: {len(all_chars-dict_chars)}\n {all_chars-dict_chars}')

    legal_chars = all_chars.copy()
    gen_punc = [chr(i) for i in range(gen_punc_st, gen_punc_en+1)]
    curr = ['₹'] + [chr(i) for i in range(8352, 8368)]
    legal_chars.update(set(list(string.digits+string.punctuation)+gen_punc+curr))
    legal_chars.update(' ')

    print('Total dict characters: ',len(dict_chars), legal_chars)
    print(f'Illegal characters: {len(legal_chars.difference(all_chars.union(dict_chars)))}\n {legal_chars.difference(all_chars.union(dict_chars))}')

    permissible_lines = [] # only to include speech transcript
    if args.use_external_corpus:
        permissible_lines = open(clean_dump,'r').readlines()
    sp_files = glob.glob(args.transcript+'/*.wrd') #+ glob.glob(spf2+'/*.wrd')
    lex_data = []
    for f in sp_files:
        txt = open(f,'r').read().strip().split('\n')
        for t in txt:
            lex_data.extend(t.split())
        permissible_lines.extend(txt)

    if args.test_transcript is not None:
        txt = open(args.test_transcript,'r').read().strip().split('\n')
        for t in txt:
            lex_data.extend(t.split())

    dict_chars.update(' ')
    print(lex_data)
    illegal_chars = ''.join(legal_chars.difference(all_chars.union(dict_chars))) # changing dict_chars to union of all_chars here!
    regex = re.compile('[%s]' % re.escape(illegal_chars))

    cleaned_lines = []
    for l in tqdm(permissible_lines):
        l = l.strip()
        chars = set(list(l))
        if len(chars - legal_chars) != 0:
            print(chars - legal_chars)
            continue

        space_rem = re.sub(regex,"",l)
        space_rem = re.sub(r"https?://\S+", "", space_rem)
        space_rem = re.sub(r"<.*?>", "", space_rem)
        cleaned_lines.append(space_rem)
    
    with open(clean_toks, 'w') as f:
        f.write("\n".join(cleaned_lines))

    if args.top_k != -1:
        words = list(itertools.chain(*[line.split() for line in tqdm(cleaned_lines)]))
        print(f"\nTotal Sentences after cleaning:\t{len(cleaned_lines)}")
        counter = Counter(words)
        total_words = sum(counter.values())
        top_counter = counter.most_common(args.top_k)
        top_counter_list = [word for word, _ in top_counter]
        ctr=[count for word, count in top_counter]
        top_words_sum = sum(ctr)
        print(f'Fraction coverage, {top_words_sum/total_words}, \tMinimum frequency, {min(ctr)}')
        new_lex = set(lex_data + top_counter_list)
    else:
        new_lex = set(lex_data)
    print(f'Length of new lexicon={len(new_lex)}')
    with open(lexicon_kenlm, 'w') as f:
        f.write("\n".join(new_lex))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate lm_data and top-k vocab."
    )
    parser.add_argument(
        "-d",
        "--dir_path",
        help="Path to the lm directory",
        type=str,
        default='',
    )
    parser.add_argument(
        "-l",
        "--lang", help="language_dataset name, eg: tamil, telugu, etc", type=str, required=True
    )
    parser.add_argument(
        "--transcript", help="path to speech transcript file", type=str, required=True
    )
    parser.add_argument("--use_external_corpus", type=bool, default=True)
    parser.add_argument("--st", type=int, required=True)
    parser.add_argument("--en", type=int, required=True)
    parser.add_argument(
        "--top_k",
        help="Use top_k most frequent words in the external corpus. If -1, only transcript lexicon will be used",
        type=int,
        default=-1
    )
    parser.add_argument("--test_transcript",  help="path to test transcript", type=str)
    args = parser.parse_args()
    main(args)

