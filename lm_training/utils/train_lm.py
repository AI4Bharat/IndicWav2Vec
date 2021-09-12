#This code has been taken from https://github.com/mozilla/DeepSpeech/blob/master/data/lm/generate_lm.py

import argparse
import os
import subprocess
from tqdm import tqdm
import shutil

def convert_and_filter_topk(args):
    """ Convert to lowercase, count word occurrences and save top-k words to a file """

    lexicon_in_path = os.path.join(args.lm_dir, 'lexicon.txt')
    lexicon_out_path = os.path.join(args.lm_dir, 'lexicon.lst')
    data_lower = shutil.copyfile(args.input_txt, os.path.join(args.lm_dir, "lower.txt.gz"))

    with open(lexicon_in_path) as f:
        content_list = f.readlines()
    content_list = [i.replace("\n", "") for i in content_list]
    first = True
    vocab_lst = []
    with open(lexicon_out_path, "w+") as f:
        for line in tqdm(content_list):
            line =line.split('\t')[0]
            vocab_lst.append(line)
            if first:
                print(line)
                first = False
            print(str(line + "\t" + " ".join(list(line.replace("/n", "").replace(" ", "|").strip())) + " |"), file=f)
    vocab_str = "\n".join(vocab_lst)

    return data_lower, vocab_str


def build_lm(args, data_lower, vocab_str):
    print("\nCreating ARPA file ...")
    lm_path = os.path.join(args.lm_dir, "lm.arpa")
    subargs = [
            os.path.join(args.kenlm_bins, "lmplz"),
            "--order",
            str(args.arpa_order),
            "--temp_prefix",
            args.lm_dir,
            "--memory",
            args.max_arpa_memory,
            "--text",
            data_lower,
            "--arpa",
            lm_path,
            "--prune",
            *args.arpa_prune.split("|"),
            "--discount_fallback",
        ]
    if args.discount_fallback:
        subargs += ["--discount_fallback"]
    subprocess.check_call(subargs)

    # Filter LM using vocabulary of top-k words
    print("\nFiltering ARPA file using vocabulary of top-k words ...")
    filtered_path = os.path.join(args.lm_dir, "lm_filtered.arpa")
    subprocess.run(
        [
            os.path.join(args.kenlm_bins, "filter"),
            "single",
            "model:{}".format(lm_path),
            filtered_path,
        ],
        input=vocab_str.encode("utf-8"),
        check=True,
    )

    # Quantize and produce trie binary.
    print("\nBuilding lm.binary ...")
    binary_path = os.path.join(args.lm_dir, "lm.binary")
    subprocess.check_call(
        [
            os.path.join(args.kenlm_bins, "build_binary"),
            # "-a",
            # str(args.binary_a_bits),
            # "-q",
            # str(args.binary_q_bits),
            # "-v",
            # args.binary_type,
            filtered_path,
            binary_path,
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate lm.binary and lexicon.lst."
    )
    parser.add_argument(
        "--input_txt",
        help="Path to a file.txt or file.txt.gz with sample sentences",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lm_dir", help="Directory path for the lm", type=str, required=True
    )
    parser.add_argument(
        "--top_k",
        help="Use top_k most frequent words for the vocab.txt file. These will be used to filter the ARPA file.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--lexicon",
        help="Use top_k most frequent words for the lexicon.txt file. These will be used to create lexicon.lst file.",
        type=int,
        required=True,
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
        "--discount_fallback",
        help="To try when such message is returned by kenlm: 'Could not calculate Kneser-Ney discounts [...] rerun with --discount_fallback'",
        action="store_true",
    )

    args = parser.parse_args()
    data_lower, vocab_str = convert_and_filter_topk(args)
    build_lm(args, data_lower, vocab_str)

    # Delete intermediate files
    os.remove(os.path.join(args.lm_dir, "lower.txt.gz"))
    os.remove(os.path.join(args.lm_dir, "lm.arpa"))
    os.remove(os.path.join(args.lm_dir, "lm_filtered.arpa"))


if __name__ == "__main__":
    main()
