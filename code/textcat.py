#!/usr/bin/env python3
"""
Computes the log probability of the sequence of tokens in file,
according to a trigram model.  The training source is specified by
the currently open corpus, and the smoothing method used by
prob() is polymorphic.
"""
import argparse
import logging
from pathlib import Path
import math

try:
    # Numpy is your friend. Not *using* it will make your program so slow.
    # So if you comment this block out instead of dealing with it, you're
    # making your own life worse.
    #
    # We made this easier by including the environment file in this folder.
    # Install Miniconda, then create and activate the provided environment.
    import numpy as np
except ImportError:
    print("\nERROR! Try installing Miniconda and activating it.\n")
    raise


from Probs import LanguageModel

TRAIN = "TRAIN"
TEST = "TEST"

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def get_model_filename(smoother: str, lexicon: Path, train_file: Path) -> Path:
    return Path(f"{smoother}_{lexicon.name}_{train_file.name}.model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("mode", choices={TRAIN, TEST}, help="execution mode")
    parser.add_argument(
        "smoother",
        type=str,
        help="""Possible values: uniform, add1, backoff_add1, backoff_wb, loglinear1
  (the "1" in add1/backoff_add1 can be replaced with any real λ >= 0
   the "1" in loglinear1 can be replaced with any C >= 0 )
""",
    )
    parser.add_argument(
        "lexicon",
        type=Path,
        help="location of the word vector file; only used in the loglinear model",
    )
    parser.add_argument("train_file1", type=Path, help="location of the training corpus")
    parser.add_argument("train_file2", type=Path, help="location of the training corpus")
    parser.add_argument("test_files", type=Path, nargs="*")

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    args = parser.parse_args()

    # Sanity-check the configuration.
    if args.mode == "TRAIN" and args.test_files:
        parser.error("Shouldn't see test files when training.")
    elif args.mode == "TEST" and not args.test_files:
        parser.error("No test files specified.")

    return args


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)
    model_path_1 = get_model_filename(args.smoother, args.lexicon, args.train_file1)
    model_path_2 = get_model_filename(args.smoother, args.lexicon, args.train_file2)
    if args.mode == TRAIN:
        log.info("Training...")
        lm_1 = LanguageModel.make(args.smoother, args.lexicon)
        lm_2 = LanguageModel.make(args.smoother, args.lexicon)
        lm_1.set_vocab_size(args.train_file1, args.train_file2)
        lm_2.set_vocab_size(args.train_file1, args.train_file2)
        lm_1.train(args.train_file1)
        lm_2.train(args.train_file2)
        lm_1.save(destination=model_path_1)
        lm_2.save(destination=model_path_2)

    elif args.mode == TEST:
        log.info("Testing...")

        train_file_1 = str(model_path_1).split('_')[2]
        type_1 = train_file_1.split('.')[0]
        train_file_2 = str(model_path_2).split('_')[2]
        type_2 = train_file_2.split('.')[0]

        lm_1 = LanguageModel.load(model_path_1)
        lm_2 = LanguageModel.load(model_path_2)
        # We use natural log for our internal computations and that's
        # the kind of log-probability that fileLogProb returns.
        # But we'd like to print a value in bits: so we convert
        # log base e to log base 2 at print time, by dividing by log(2).

        log.info("Printing file log-likelihoods.")

        # total_log_prob = 0.0
        prior = float(str(args.test_files[0]))
        # prior = 10**(-40)
        num_files = len(args.test_files)
        num_gen, num_spam = 0, 0
        for idx, test_file in enumerate(args.test_files):
            if idx == 0:
                continue
            log_prob_1 = lm_1.file_log_prob(test_file)
            log_prob_2 = lm_2.file_log_prob(test_file)
            post_1 = prior*math.exp(log_prob_1)
            post_2 = (1-prior)*math.exp(log_prob_2)
            # print("gen log prob: " + str(math.exp(log_prob_1)))
            # print("gen log prob: " + str(math.exp(log_prob_2)))

            if post_1 > post_2:
                print(type_1 + " " + str(test_file))
                num_gen += 1
            else:
                print(type_2 + " " + str(test_file))
                num_spam += 1
        print(str(num_gen) + " files were more probably " + type_1 + " (" + str(num_gen/num_files) + ")")
        print(str(num_spam) + " files were more probably " + type_2 + " (" + str(num_spam / num_files) + ")")
    else:
        raise ValueError("Inappropriate mode of operation.")


if __name__ == "__main__":
    main()

