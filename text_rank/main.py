import os
import json
import argparse

from text_rank.summarize import summarize
from text_rank.keywords import keywords


def generate_output_name(input_name, task):
    path, f = os.path.split(input_name)
    f, ext = os.path.splitext(f)
    return os.path.join(path, f"{f}-{task}{ext}")


def main():
    parser = argparse.ArgumentParser("Run Text Rank")
    parser.add_argument("files", nargs="+", help="A list of files that contain documents to process.")
    parser.add_argument(
        "--task",
        default="summarize",
        choices={"summarize", "keywords"},
        help="Summarize the document or extract keywords?",
    )
    parser.add_argument(
        "--count", default=5, type=int, help="The number of sentences/keywords to return",
    )
    parser.add_argument(
        "--iters", "-i", type=int, default=20, help="The maximum number of iterations to run text rank for.",
    )
    parser.add_argument(
        "--convergence",
        "-c",
        type=float,
        default=0.0001,
        help="The threshold of differences between iterations that causes the algorithm to stop.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show progress bars when fitting text rank.",
    )
    parser.add_argument("--indent", default=2, type=int, help="Indent formatting for the output json")
    args = parser.parse_args()

    process = summarize if args.task == "summarize" else keywords
    file_ext = "summary" if args.task == "summarize" else "keywords"

    for file_name in args.files:
        with open(file_name) as f:
            document = json.load(f)
        if not isinstance(document, list):
            raise ValueError(f"Data should be a list, found: {type(document)}")
        if args.task == "summarize" and not all(isinstance(sent, str) for sent in document):
            raise ValueError(f"Data should be a list of strings")
        if args.task == "keywords" and not all(isinstance(token, dict) for token in document):
            raise ValueError(f"Data should be a list of dictionaries")

        result = list(process(document, args.count))

        output = generate_output_name(file_name, file_ext)
        with open(output, "w") as wf:
            json.dump(result, wf, indent=2)


if __name__ == "__main__":
    main()
