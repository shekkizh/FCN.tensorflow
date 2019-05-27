import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import argparse
import os

parser = argparse.ArgumentParser(prog="It's a cool job", usage="I can use it by this", description="description")
parser.add_argument("--folder", type=str, default="./visualize_results", help="help content")

args = parser.parse_args()
file_dir = args.folder

def read_folder():
    inp_folder = []
    gt_folder = []
    pred_folder = []
    for item in os.listdir(file_dir):
        if item.split("_")[0] == "inp":
            inp_folder.append(item)
        elif item.split("_")[0] == "gt":
            gt_folder.append(item)
        else:
            pred_folder.append(item)
    image_list = map(sorted, [inp_folder, gt_folder, pred_folder])
    for item in zip(*image_list):
        draw_heatmap(item[1], item[0], item[2])


def draw_heatmap(inp, gt, pred):
    inp_file = os.path.join(file_dir, inp)
    gt_file = os.path.join(file_dir, gt)
    pred_file = os.path.join(file_dir, pred)
    gt = misc.imread(inp_file)
    inp = misc.imread(gt_file)
    pred = misc.imread(pred_file)
    plt.figure(1)
    plt.subplot(131)
    plt.title("inp image")
    plt.imshow(inp)
    plt.subplot(132)
    plt.title("gt")
    plt.imshow(gt, cmap=cm.Paired, vmin=0, vmax=151)
    plt.axis("off")
    plt.subplot(133)
    plt.title("pred")
    plt.imshow(pred, cmap=cm.Paired, vmin=0, vmax=151)
    plt.axis("off")
    file_id = os.path.basename(inp_file).split(".")[0].split("_")[1]
    plt.savefig(args.folder+"/heatmap_"+file_id+".png")


if __name__ == "__main__":
    read_folder()
