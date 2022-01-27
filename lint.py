""" python lint.py -p path -t t lints files at `path` with passing threshold of `t` (0-10) """
import argparse
import logging
from pylint.lint import Run


def main():
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(prog="LINT")

    parser.add_argument(
        "-p",
        "--path",
        help="path to directory you want to run pylint | "
        "Default: %(default)s | "
        "Type: %(type)s ",
        default="./tidy3d",
        type=str,
    )

    parser.add_argument(
        "-t",
        "--threshold",
        help="score threshold to fail pylint runner | " "Default: %(default)s | " "Type: %(type)s ",
        default=9.0,
        type=float,
    )

    args = parser.parse_args()
    path = str(args.path)
    threshold = float(args.threshold)

    logging.info("PyLint Starting | " "Path: {} | " "Threshold: {} ".format(path, threshold))

    results = Run([path], do_exit=False)

    final_score = results.linter.stats.global_note

    if final_score < threshold:

        message = "PyLint Failed | " "Score: {} | " "Threshold: {} ".format(final_score, threshold)

        logging.error(message)
        raise Exception(message)

    else:
        message = "PyLint Passed | " "Score: {} | " "Threshold: {} ".format(final_score, threshold)

        logging.info(message)

        exit(0)


if __name__ == "__main__":
    main()
