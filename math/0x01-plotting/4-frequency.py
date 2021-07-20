#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.xlim(0, 100)
plt.ylim(0, 30)
limits = np.arange(0, 110, 10)
y_limits = np.arange(0, 30, 5)
plt.xticks(limits)
plt.yticks(y_limits)
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.hist(student_grades, bins=limits, edgecolor='black', linewidth=1.2)

plt.show()
