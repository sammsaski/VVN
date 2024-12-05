import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

x = ['1/255', '2/255', '3/255']

zoomin_4 = [21.41, 22.25, 23.00]
zoomin_8 = [62.03, 63.53, 64.81]
zoomin_16 = [204.42, 211.56, 218.68]

zoomout_4 = [22.79, 32.54, 39.31]
zoomout_8 = [63.64, 125.74, 129.30]
zoomout_16 = [206.37, 214.54, 223.02]

gtsrb_4 = [6.31, 7.60, 9.23]
gtsrb_8 = [20.59, 21.62, 23.74]
gtsrb_16 = [197.03, 426.74, 948.58]

stmnist_16 = [22.35, 27.81, 35.06]
stmnist_32 = [13.06, 15.05, 20.27]
stmnist_64 = [6.76, 7.91, 300.18]


if __name__ == "__main__":
    plt.figure(figsize=(7, 8))

    plt.plot(x, zoomin_4, linestyle='solid', color='red')
    plt.plot(x, zoomin_8, linestyle='solid', color='green')
    plt.plot(x, zoomin_16, linestyle='solid', color='blue')

    plt.plot(x, zoomout_4, linestyle='dashed', color='red')
    plt.plot(x, zoomout_8, linestyle='dashed', color='green')
    plt.plot(x, zoomout_16, linestyle='dashed', color='blue')

    plt.plot(x, gtsrb_4, linestyle='dotted', color='red')
    plt.plot(x, gtsrb_8, linestyle='dotted', color='green')
    plt.plot(x, gtsrb_16, linestyle='dotted', color='blue')

    # plt.plot(x, stmnist_16, linestyle='dashdot', color='red')
    # plt.plot(x, stmnist_32, linestyle='dashdot', color='green')
    # plt.plot(x, stmnist_64, linestyle='dashdot', color='blue')

    plt.yscale('log')

    # legend lines
    line1 = mlines.Line2D([], [], color='black', linestyle='solid', label='Zoom In')
    line2 = mlines.Line2D([], [], color='black', linestyle='dashed', label='Zoom Out')
    line3 = mlines.Line2D([], [], color='black', linestyle='dotted', label='GTSRB')

    # legend colors
    color_red = mpatches.Patch(color='red', label='4 frames')
    color_green = mpatches.Patch(color='green', label='8 frames')
    color_blue = mpatches.Patch(color='blue', label='16 frames')

    # add labels and title
    plt.xlabel("Epsilon")
    plt.ylabel("Runtime (s)")
    plt.title("Comparison of average runtime")
    plt.legend(handles=[line1, line2, line3, color_red, color_green, color_blue], loc='upper left')


    plt.savefig('runtime_comp.png', dpi=300)
