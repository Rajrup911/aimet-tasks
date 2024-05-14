import argparse
import os
import json
from infer import validation

def arguments(raw_args):
    parser = argparse.ArgumentParser(description='Create')
    parser.add_argument('--config', help='model configuration to use', type=str, required=True)
    args = parser.parse_args(raw_args)
    return args


def main(raw_args=None):
    args = arguments(raw_args)
    if os.path.exists(args.config):
        with open(args.config) as f_in:
            config = json.load(f_in)

    validation(config['dataset'], config["Result"], config['pretrained_model'], config)
 
if __name__ == '__main__':
    main()
