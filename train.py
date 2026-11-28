"""Train a dataset-specific CNN registration model."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--architecture", choices=("baseline", "dlir"), default="baseline")
    args, remaining = parser.parse_known_args()
    if args.architecture == "dlir":
        from cnn.dlir_training import main as train
    else:
        from cnn.training import main as train
    train(remaining)


if __name__ == "__main__":
    main()
