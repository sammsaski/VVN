import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
import sys
import json
from collections import defaultdict


with open("src/analysis/final_results.json", "r") as json_file:
    data = json.load(json_file)

    zoomout = data["ScalabilityZoomOut"]["per_frame_time"]
    zoomin = data["ScalabilityZoomIn"]["per_frame_time"]

    x_data = list(range(1, 9))

    plt.xlabel("Frame Number")
    plt.ylabel("Avg. Time (s)")

    plt.plot(x_data, zoomout["epsilon_1"], color="red", label="ε=1/255 *")
    plt.plot(
        x_data, zoomin["epsilon_1"], linestyle="--", color="red", label="ε=1/255 ^"
    )

    plt.plot(x_data, zoomout["epsilon_2"], color="blue", label="ε=2/255 *")
    plt.plot(
        x_data, zoomin["epsilon_2"], linestyle="--", color="blue", label="ε=2/255 ^"
    )

    plt.plot(x_data, zoomout["epsilon_3"], color="green", label="ε=3/255 *")
    plt.plot(
        x_data, zoomin["epsilon_3"], linestyle="--", color="green", label="ε=3/255 ^"
    )

    plt.grid(True)
    plt.legend(bbox_to_anchor=(0.5, 1.17), loc="upper center", ncol=3)
    plt.show()
