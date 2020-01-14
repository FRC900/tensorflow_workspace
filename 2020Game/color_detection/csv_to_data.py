import numpy as np

data = []
target = []
n = 300
with open("compiled_blue.csv") as f:
    m = 0
    data_lines = f.readlines()[1:]
    for line in data_lines:
        arr = np.array(list(map(int, line.split(","))))
        data.append(arr/max(arr))
        target.append([0,0,0,1])
        m+=1
        if(m >= n):
            break

with open("compiled_red.csv") as f:
    m = 0
    data_lines = f.readlines()[1:]
    for line in data_lines:
        arr = np.array(list(map(int, line.split(","))))
        data.append(arr/max(arr))
        target.append([1,0,0,0])
        m+=1
        if(m >= n):
            break

with open("compiled_yellow.csv") as f:
    m = 0
    data_lines = f.readlines()[1:]
    for line in data_lines:
        arr = np.array(list(map(int, line.split(","))))
        data.append(arr/max(arr))
        target.append([0,0,1,0])
        m+=1
        if(m >= n):
            break

with open("compiled_green.csv") as f:
    m = 0
    data_lines = f.readlines()[1:]
    for line in data_lines:
        arr = np.array(list(map(int, line.split(","))))
        data.append(arr/max(arr))
        target.append([0,1,0,0])
        m+=1
        if(m >= n):
            break

np.savez_compressed("data.npz", X=np.array(data), Y=np.array(target))