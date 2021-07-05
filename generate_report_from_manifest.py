# taken from https://github.com/Open-Speech-EkStep/vakyansh-wav2vec2-experimentation/blob/main/utils/analysis/generate_wav_report_from_tsv.py
import pandas as pd
import numpy as np
import argparse


def generate_report(tsv_file):

    df = pd.read_csv(tsv_file, sep="\t", header=0, names=["file", "frames"])
    df["duration_in_sec"] = [i / 16000 for i in df["frames"]]

    print("=" * 100)
    print("Total number of files in ", tsv_file, ":", len(df))
    total_duration = sum(df["duration_in_sec"]) / 3600
    print("Total duration in hours: ", total_duration)

    min_duration = min(df["duration_in_sec"])
    max_duration = max(df["duration_in_sec"])

    print("Min duration of a file in seconds:", min_duration)
    print("Max duration of a file in seconds:", max_duration)

    mean_duration = np.mean(df["duration_in_sec"])
    print("Mean duration across files in seconds:", mean_duration)

    files_less_than_equal_to_1s = []
    files_less_than_equal_to_5s = []
    min_dur = []

    for index, dur in enumerate(df["duration_in_sec"]):
        if dur <= 1:
            files_less_than_equal_to_1s.append(dur)
        if dur <= 5:
            files_less_than_equal_to_5s.append(dur)

    print(
        "Number of files with duration less than/equal to 1 seconds:",
        len(files_less_than_equal_to_1s),
    )
    print(
        "Total duration of files less thanthan/equal to 1 seconds:",
        sum(files_less_than_equal_to_1s) / 3600,
        " hours",
    )
    print("*" * 100)
    print(
        "Number of files with duration less than/equal to 5 seconds:",
        len(files_less_than_equal_to_5s),
    )
    print(
        "Total duration of files less thanthan/equal to 5 seconds:",
        sum(files_less_than_equal_to_5s) / 3600,
        " hours",
    )

    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument("--tsv", type=str, help="tsv file to analyse")
    args = parser.parse_args()
    tsv_file = args.tsv
    generate_report(tsv_file)