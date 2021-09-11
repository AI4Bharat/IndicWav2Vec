## Authors: Aman_Tiwari, Harveen Chadha
## Usage: python manifest.py <path to wav files> --dest <path to destination tsv> --ext wav --train-name train.tsv --valid-percent 0 --jobs -1

# taken from https://github.com/Open-Speech-EkStep/common_scripts/blob/main/prep_scripts/manifest.py

import argparse
import glob
import os
import random
from joblib import Parallel, delayed
import soundfile
from random import sample
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root", metavar="DIR", help="root directory containing flac files to index"
    )
    parser.add_argument(
        "--valid-percent",
        default=0.01,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--dest", default=".", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--train-name", default="train", type=str, help="name of Training TSV file"
    )
    parser.add_argument(
        "--valid-name", default="valid", type=str, help="name of Validation TSV file"
    )
    parser.add_argument(
        "--jobs", default=-1, type=int, help="Number of jobs to run in parallel"
    )

    parser.add_argument(
        "--ext", default="flac", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    return parser


def read_file(fname, args, dir_path):
    file_path = os.path.realpath(fname)

    if args.path_must_contain and args.path_must_contain not in file_path:
        pass

    frames = soundfile.info(fname).frames
    if (
        frames > 0 and frames <= 480000
    ):  ## Added to remove any audio greater than 30 seconds
        ret_val = "{}\t{}\n".format(os.path.relpath(file_path, dir_path), frames)
        return ret_val
    else:
        return ""


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "**/*." + args.ext)
    # rand = random.Random(args.seed)

    random.seed(args.seed)

    local_arr = []

    local_arr.extend(
        Parallel(n_jobs=args.jobs)(
            delayed(read_file)(fname, args, dir_path)
            for fname in tqdm(glob.iglob(search_path, recursive=True))
        )
    )

    # dest = train_f if rand.random() > args.valid_percent else valid_f

    valid_samples = sample(local_arr, int(len(local_arr) * float(args.valid_percent)))
    train_samples = list(set(local_arr) - set(valid_samples))

    ## Include the directory at top of manifest file
    train_samples.insert(0, str(dir_path).strip() + "\n")
    valid_samples.insert(0, str(dir_path).strip() + "\n")

    ## Saving Training Manifest File
    with open(os.path.join(args.dest, args.train_name + ".tsv"), "w+") as train_f:
        train_f.writelines(train_samples)
        print(
            "** Writing Training Manifest File done with ",
            len(train_samples),
            " records",
        )

    ## Saving Validation Manifest File only if it is to be made
    if len(valid_samples) != 1:
        with open(os.path.join(args.dest, args.valid_name + ".tsv"), "w+") as valid_f:
            valid_f.writelines(valid_samples)
            print(
                "** Writing Validation Manifest File done with ",
                len(valid_samples),
                " records",
            )

    print("## Manifest Creation Done")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)