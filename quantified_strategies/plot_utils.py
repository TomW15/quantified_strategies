import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

from . import utils

style.use("ggplot")


def create_loss_graph(model_name: str) -> None:

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    color_map = {"train": colors[0], "test": colors[1]}

    PATH = utils.get_path(path=__file__)
    contents = pd.read_csv(PATH / f"scripts/outputs/models/{model_name}.log", names=["name", "time", "epoch", "loss", "hit_rate", "val_loss", "val_hit_rate"])

    fig = plt.figure(figsize=(15, 7))

    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax2 = plt.subplot2grid((2, 1), (1, 0), sharex=ax1)

    ax1.plot(contents["epoch"], contents["loss"], label="loss", color=color_map["train"])
    # ax1.set_yscale("log")
    ax1.legend(loc=2)
    twinx1 = ax1#.twinx()
    twinx1.plot(contents["epoch"], contents["val_loss"], label="val_loss", color=color_map["test"])
    # twinx1.set_yscale("log")
    twinx1.legend(loc=1)

    ax2.plot(contents["epoch"], contents["hit_rate"], label="hit_rate", color=color_map["train"])
    twinx2 = ax2#.twinx()
    twinx2.plot(contents["epoch"], contents["val_hit_rate"], label="val_hit_rate", color=color_map["test"])
    ax2.legend(loc=2)

    n = contents.shape[0] // 8
    ax1.set_xticklabels(contents["epoch"].tolist()[::n])
    ax1.set_xticks([*range(0, contents.shape[0], n)])

    plt.show()

    return
