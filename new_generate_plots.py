import matplotlib.pyplot as plt

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
relax_zi_4f = [21.41, 22.25, 23.00]
relax_zi_8f = [62.03, 63.53, 64.81]
approx_zo_4f = [41.86, 59.99, 72.33]
approx_zo_8f = [115.43, 227.01, 234.41]
approx_zi_4f = [39.58, 40.99, 42.44]
approx_zi_8f = [111.36, 113.87, 116.58]

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
plt.xlabel('epsilon')
plt.ylabel('Runtime (s)')

plt.legend()

plt.savefig('figs/relax_approx/runtime_comp.png')
