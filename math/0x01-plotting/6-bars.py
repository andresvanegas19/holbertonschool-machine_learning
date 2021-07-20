#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

bars = np.arange(len(fruit[0]))
name = ["Farrah", "Fred", "Felicia"]
plt.bar(bars, fruit[0], width=0.3, color='red')
plt.bar(bars, fruit[1], width=0.3, bottom=np.array(fruit[0]), color='yellow')
plt.bar(bars, fruit[2], width=0.3, bottom=np.array(
    fruit[0])+np.array(fruit[1]), color='#ff8000')
plt.bar(bars, fruit[3], width=0.3, bottom=np.array(fruit[0]) +
        np.array(fruit[1])+np.array(fruit[2]), color='#ffe5b4')

plt.title('Number of Fruit per Person')
plt.ylabel('Quantity of Fruit')
plt.xticks(np.arange(len(name)), name, size="xx-large")
plt.yticks(np.arange(0, 90, 10))
plt.legend(name)


plt.show()
