import matplotlib.pyplot as plt
import numpy as np

##############
# BAR GRAPHS #
##############

# 1. First, relax vs. approx comparison
# Data for bar graphs
categories = ['relax_zo_4f', 'relax_zo_8f', 'relax_zi_4f', 'relax_zi_8f', 'approx_zo_4f', 'approx_zo_8f', 'approx_zi_4f', 'approx_zi_8f']
values_eps1 = [97, 100, 98, 100, 97, 100, 98, 100]
values_eps2 = [96, 98, 94, 96, 98, 96, 94, 96]
values_eps3 = [93, 97, 93, 93, 93, 97, 93, 93]

# Data for line charts
# t_values_eps1 = [22.79, 63.64, 21.41, 62.03, 41.86, 115.43, 39.58, 111.36]
# t_values_eps2 = [32.54, 125.74, 22.25, 63.53, 59.99, 227.01, 40.99, 113.87]
# t_values_eps3 = [39.31, 129.3, 23.00, 64.81, 72.33, 234.41, 42.44, 116.58]

x = ['1/255', '2/255', '3/255']

relax_zo_4f = [22.79, 32.54, 39.31]
relax_zo_8f = [63.64, 125.74, 129.3]
relax_zo_16f = [206.37, 214.54, 223.02]
relax_zi_4f = [21.41, 22.25, 23.00]
relax_zi_8f = [62.03, 63.53, 64.81]
relax_zi_16f = [204.42, 211.56, 218.68]
approx_zo_4f = [41.86, 59.99, 72.33]
approx_zo_8f = [115.43, 227.01, 234.41]
approx_zi_4f = [39.58, 40.99, 42.44]
approx_zi_8f = [111.36, 113.87, 116.58]

"""
plt.figure(figsize=(10, 6))
plt.plot(x, relax_zo_4f, marker='o', linestyle='-', color='red', label='ZoomOut-4f')
plt.plot(x, relax_zo_8f, marker='o', linestyle='-', color='red', label='ZoomOut-8f')
plt.plot(x, relax_zi_4f, marker='^', linestyle='-', color='red', label='ZoomIn-4f')
plt.plot(x, relax_zi_8f, marker='^', linestyle='-', color='red', label='ZoomIn-8f')
plt.plot(x, approx_zo_4f, marker='o', linestyle='-', color='blue', label='ZoomOut-4f')
plt.plot(x, approx_zo_8f, marker='o', linestyle='-', color='blue', label='ZoomOut-8f')
plt.plot(x, approx_zi_4f, marker='^', linestyle='-', color='blue', label='ZoomIn-4f')
plt.plot(x, approx_zi_8f, marker='^', linestyle='-', color='blue', label='ZoomIn-8f')

plt.title('Runtime comparison of relax vs. approx verification')
plt.xlabel('Epsilon')
plt.ylabel('Average runtime (s)')

plt.legend()

plt.savefig('figs/relax_approx/runtime_comp.png')
"""

################
# 2. Runtime comparison for change in dataset length
plt.figure(figsize=(10, 6))
categories = ['4', '8', '16']
subcategories = ['1/255', '2/255', '3/255']

zo_values = [
    relax_zo_4f,
    relax_zo_8f,
    relax_zo_16f
]

zi_values = [
    relax_zi_4f,
    relax_zi_8f,
    relax_zi_16f
]

n_categories = len(categories)
n_subcategories = len(x)

x = np.arange(n_categories)
bar_width = 0.2

# ZOOM IN
fig, ax = plt.subplots(figsize=(10,6))

for i in range(n_subcategories):
    ax.bar(x + i * bar_width, [value[i] for value in zi_values], width=bar_width, label=subcategories[i])

ax.set_title('Runtime comparison when varying lengths of ZoomIn datasets.')
ax.set_xlabel('Length of videos')
ax.set_ylabel('Avg. Runtime (s)')

ax.set_xticks(x + bar_width * (n_subcategories - 1) / 2)
ax.set_xticklabels(categories)

ax.legend(title='Epsilon')

plt.savefig('figs/length_of_video/zoom_in_comp.png')


# ZOOM OUT
fig, ax = plt.subplots(figsize=(10,6))

for i in range(n_subcategories):
    ax.bar(x + i * bar_width, [value[i] for value in zo_values], width=bar_width, label=subcategories[i])

ax.set_title('Runtime comparison when varying lengths of ZoomOut datasets.')
ax.set_xlabel('Length of videos')
ax.set_ylabel('Avg. Runtime (s)')

ax.set_xticks(x + bar_width * (n_subcategories - 1) / 2)
ax.set_xticklabels(categories)

ax.legend(title='Epsilon')

plt.savefig('figs/length_of_video/zoom_out_comp.png')
