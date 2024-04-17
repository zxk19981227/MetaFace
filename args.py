import argparse


def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cfg',type=str,default='yml/vocaset.yaml')
    args=parser.parse_args()
    return args