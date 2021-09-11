import jiwer
import jiwer.transforms as tr

from jiwer import compute_measures
from typing import List


def compute_wer(predictions=None, references=None, concatenate_texts=False):
    if concatenate_texts:
        return compute_measures(references, predictions)
    else:
        incorrect = 0
        total = 0
        for prediction, reference in zip(predictions, references):
            measures = compute_measures(reference, prediction)
            incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
            total += measures["substitutions"] + measures["deletions"] + measures["hits"]
        return measures


class SentencesToListOfCharacters(tr.AbstractTransform):

    def process_string(self,s):
        return list(s)

    def process_list(self, inp: List[str]):
        chars = []

        for sentence in inp:
            chars.extend(self.process_string(sentence))

        return chars


cer_transform = tr.Compose(
    [
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
        SentencesToListOfCharacters(),
    ]
)


def compute_cer(predictions, references, concatenate_texts=False):

    if concatenate_texts:
        return jiwer.wer(
            references,
            predictions,
            truth_transform=cer_transform,
            hypothesis_transform=cer_transform,
        )

    incorrect = 0
    total = 0
    for prediction, reference in zip(predictions, references):
        measures = jiwer.compute_measures(
            reference,
            prediction,
            truth_transform=cer_transform,
            hypothesis_transform=cer_transform,
        )
        incorrect += measures["substitutions"] + measures["deletions"] + measures["insertions"]
        total += measures["substitutions"] + measures["deletions"] + measures["hits"]

    return incorrect / total


if __name__ == "__main__":
    print(compute_wer(['my name is'],['my name']))
