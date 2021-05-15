import argparse

import yaml

from preprocessor import ljspeech


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_data(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
