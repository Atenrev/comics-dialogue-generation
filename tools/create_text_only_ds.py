import argparse
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="dataset/COMICS_ocr_file copy.csv",
                        help='dataset path')
    parser.add_argument('--context_panels', type=int, default=3,
                        help='context_panels')
    parser.add_argument('--context_max_boxes_size', type=int, default=3,
                        help='context_max_boxes_size')
    parser.add_argument('--context_max_speech_size', type=int, default=30,
                        help='context_max_speech_size')
    parser.add_argument('--answer_candidates', type=int, default=3,
                        help='answer_candidates')
    parser.add_argument('--answer_max_tokens', type=int, default=30,
                        help='answer_max_tokens')

    args = parser.parse_args()
    return args


def main(args: argparse.Namespace) -> None:
    # TODO: Preprocess with bounding boxes on context and target answer
    df = pd.read_csv(args.dataset, ',')
    df = df.dropna()
    df_grouped = df.groupby(['comic_no', 'page_no', 'panel_no'])
    pass


if __name__ == "__main__":
    args = parse_args()
    main(args)