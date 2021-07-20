#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig, ax = plt.subplots()
fig.tight_layout(pad=3)

# delete the axis for add padding to each plot
ax.axis('off')

# add the linear plot
ax = fig.add_subplot(3, 2, 1)
ax.plot(range(len(y0)), y0, color='red')
ax.set_yticks(np.arange(0, 1500, 500))
ax.set_xticks(np.arange(0, 11))

# add a scatter plot
ax1 = fig.add_subplot(3, 2, 2)
ax1.scatter(x1, y1, c="magenta")
ax1.axis(xmin=55, xmax=85, ymin=160, ymax=195)
ax1.set_yticks(np.arange(170, 200, 10))
ax1.set_xticks(np.arange(60, 90, 10))
ax1.set_xlabel('Height (in)')
ax1.set_ylabel('Weight (lbs)')
ax1.set_title('Men\'s Height vs Weight')


# add a linear plot
ax2 = fig.add_subplot(3, 2, 3)
ax2.plot(x2, y2)
ax2.set_xlim(0, 28650)
ax2.set_yscale('log')
ax2.set_xlabel('Height (in)')
ax2.set_ylabel('Weight (lbs)')
ax2.set_title('Men\'s Height vs Weight')

# add a linear a plot
ax3 = fig.add_subplot(3, 2, 4)
ax3.set_xlim(0, 20000)
ax3.set_ylim(0, 1)

ax3.set_xlabel('Time (years)')
ax3.set_ylabel('Fraction Remaining')
ax3.set_title('Exponential Decay of Radioactive Elements', fontsize=5)

ax3.plot(x3, y31, color='red', linestyle='dashed', label='C-14')
ax3.plot(x3, y32, color="green", label='Ra-226')
ax3.legend(prop={'size': 3})


# add the histogrm
ax4 = fig.add_subplot(3, 1, 3)
ax4.set_xlim(0, 100)
ax4.set_ylim(0, 30)
limits = np.arange(0, 110, 10)
y_limits = np.arange(0, 30, 5)
ax4.set_xticks(limits)
ax4.set_yticks(y_limits)
ax4.set_xlabel('Grades')
ax4.set_ylabel('Number of Students')
ax4.set_title('Project A')
ax4.hist(student_grades, bins=limits, edgecolor='black', linewidth=1.2)

plt.show()
