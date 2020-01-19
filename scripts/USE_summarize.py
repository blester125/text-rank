import os
import json
import argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow_hub as hub
from text_rank import DEMO_LOC, summarize


def cosine_sim(x, y):
    return x @ y.T / np.linalg.norm(x) * np.linalg.norm(y)


class USEEncoder:
    def __init__(self, url=None):
        url = (
            "https://tfhub.dev/google/universal-sentence-encoder/4"
            if url is None
            else url
        )
        self.embed = hub.load(url)

    def __call__(self, normed_s1, normed_s2, raw_s1, raw_s2, **kwargs):
        embedded = self.embed([raw_s1, raw_s2])
        s1, s2 = embedded.numpy()
        return cosine_sim(s1, s2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sents", default=5, type=int)
    args = parser.parse_args()

    print("=" * 80)
    print("Using the default similarity function\n")
    sentences = json.load(open(DEMO_LOC / "paper-example-summarize.json"))
    sentences = summarize(sentences, args.sents,)
    for sentence in sentences:
        print(" * " + sentence)

    print("\n" + "=" * 80)
    print("Using the Universal Sentence Encoder similarity function\n")
    use = USEEncoder()
    sentences = json.load(open(DEMO_LOC / "paper-example-summarize.json"))
    sentences = summarize(sentences, args.sents, sim=use,)
    for sentence in sentences:
        print(" * " + sentence)


if __name__ == "__main__":
    main()
