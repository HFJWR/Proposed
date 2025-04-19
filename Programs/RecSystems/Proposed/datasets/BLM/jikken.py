import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dic = {}
fontsize = 20
def read_data(path, dic):
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            [user, item] = line.strip().split(',')
            if item in dic:
                dic[item] += 1
            else:
                dic[item] = 1
read_data('train.txt', dic)
read_data('val.txt', dic)
read_data('test.txt', dic)

# plt.figure(figsize=(4,3))
plt.plot([i for i in range(len(dic))], list(sorted(dic.values(), reverse = True)), linewidth = 5)
plt.xticks([0, 60000, 120000], fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.xlabel('Item Index', fontsize = fontsize)
plt.ylabel('Number of Occurance', fontsize = fontsize)
plt.grid()
plt.tight_layout()
plt.savefig('item_occurance.pdf')
plt.show()