import os
import argparse

project_path = os.getcwd()
dataset_path = project_path + os.sep + "data" + os.sep + "songningning_bef"
save_weights_path = project_path + os.sep + "output" + os.sep + "weights.pkl"
finish_weights_path = project_path + os.sep + "checkpoints" + os.sep + "song_weights.pkl"

parser = argparse.ArgumentParser(description="Pytorch implementation of EEGNet")
parser.add_argument('--has-cuda', type=bool, default=False)
parser.add_argument('--data-path', type=str, default=dataset_path)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--num-class', type=int, default=2)
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--step-size', type=int, default=40)
parser.add_argument('--gamma', type=float, default=0.8)
parser.add_argument('--num-workers', type=int, default=0)
parser.add_argument('--print-freq', type=int, default=40)
parser.add_argument('--save-epoch-freq', type=int, default=1)
parser.add_argument('--save-weights-path', type=str, default=save_weights_path)
parser.add_argument('--finish-weights-path', type=str, default=finish_weights_path, help="Training from one checkpoint")
parser.add_argument('--start-epoch', type=int, default=0, help="Corresponding to the epoch of resume")
parser.add_argument('--network', type=str, default="EEGNet")
args = parser.parse_args()


