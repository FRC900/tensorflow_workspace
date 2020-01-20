import numpy as np


def normalize(arr):
    m = arr-np.mean(arr)
    return m/m.std()

data = []
data_straight = []
data_angle = []
target = []
target_straight = []
target_angle = []
with open("blue_straight.csv") as f:
    blue_lines = len(f.readlines())

with open("red_straight.csv") as f:
    red_lines = len(f.readlines())

with open("yellow_straight.csv") as f:
    yellow_lines = len(f.readlines())

with open("green_straight.csv") as f:
    green_lines = len(f.readlines())

n_straight = min([blue_lines, yellow_lines, red_lines, green_lines])

with open("blue_angle.csv") as f:
    blue_lines = len(f.readlines())
with open("red_angle.csv") as f:
    red_lines = len(f.readlines())
with open("yellow_angle.csv") as f:
    yellow_lines = len(f.readlines())
with open("green_angle.csv") as f:
    green_lines = len(f.readlines())

n_angle = min([blue_lines, yellow_lines, red_lines, green_lines])

with open("blue.csv") as f:
    blue_lines = len(f.readlines())
with open("red.csv") as f:
    red_lines = len(f.readlines())
with open("yellow.csv") as f:
    yellow_lines = len(f.readlines())
with open("green.csv") as f:
    green_lines = len(f.readlines())
n = min([blue_lines, red_lines, yellow_lines, green_lines])

m = 0
with open("blue_straight.csv") as f:
    data_lines = f.readlines()
    for line in data_lines:
        arr = np.array(list(map(float, line.split(","))))
        data_straight.append(normalize(arr))
        target_straight.append([1,0,0,0])
        m+=1
        if(m >= n_straight):
            break
m = 0
with open("blue_angle.csv") as f2:
    data_lines = f2.readlines()
    for line in data_lines:
        arr = np.array(list(map(float, line.split(","))))
        data_angle.append(normalize(arr))
        target_angle.append([1,0,0,0])
        m+=1
        if(m >= n_angle):
            break
m = 0
with open("red_straight.csv") as f:
    data_lines = f.readlines()
    for line in data_lines:
        arr = np.array(list(map(float, line.split(","))))
        data_straight.append(normalize(arr))
        target_straight.append([0,1,0,0])
        m+=1
        if(m >= n_straight):
            break
m = 0
with open("red_angle.csv") as f2:
    data_lines = f2.readlines()
    for line in data_lines:
        arr = np.array(list(map(float, line.split(","))))
        data_angle.append(normalize(arr))
        target_angle.append([0,1,0,0])
        m+=1
        if(m >= n_angle):
            break
m = 0
with open("yellow_straight.csv") as f:
    data_lines = f.readlines()
    for line in data_lines:
        arr = np.array(list(map(float, line.split(","))))
        data_straight.append(normalize(arr))
        target_straight.append([0,0,1,0])
        m+=1
        if(m >= n_straight):
            break
m = 0
with open("yellow_angle.csv") as f2:
    data_lines = f2.readlines()
    for line in data_lines:
        arr = np.array(list(map(float, line.split(","))))
        data_angle.append(normalize(arr))
        target_angle.append([0,0,1,0])
        m+=1
        if(m >= n_angle):
            break

m = 0
with open("green_straight.csv") as f:
    data_lines = f.readlines()
    for line in data_lines:
        arr = np.array(list(map(float, line.split(","))))
        data_straight.append(normalize(arr))
        target_straight.append([0,0,0,1])
        m+=1
        if(m >= n_straight):
            break
m = 0
with open("green_angle.csv") as f2:
    data_lines = f2.readlines()
    for line in data_lines:
        arr = np.array(list(map(float, line.split(","))))
        data_angle.append(normalize(arr))
        target_angle.append([0,0,0,1])
        m+=1
        if(m >= n_angle):
            break

np.savez_compressed("data_angle.npz", X=np.array(data_angle), Y=np.array(target_angle))
np.savez_compressed("data_straight.npz", X=np.array(data_straight), Y=np.array(target_straight))

m = 0
with open("blue.csv") as f:
    data_lines = f.readlines()
    for line in data_lines:
        arr = np.array(list(map(float, line.split(","))))
        data.append(normalize(arr))
        target.append([1,0,0,0])
        m+=1
        if(m >= n):
            break

m = 0
with open("red.csv") as f:
    data_lines = f.readlines()
    for line in data_lines:
        arr = np.array(list(map(float, line.split(","))))
        data.append(normalize(arr))
        target.append([0,1,0,0])
        m+=1
        if(m >= n):
            break

m = 0
with open("green.csv") as f:
    data_lines = f.readlines()
    for line in data_lines:
        arr = np.array(list(map(float, line.split(","))))
        data.append(normalize(arr))
        target.append([0,0,0,1])
        m+=1
        if(m >= n):
            break

m = 0
with open("yellow.csv") as f:
    data_lines = f.readlines()
    for line in data_lines:
        arr = np.array(list(map(float, line.split(","))))
        data.append(normalize(arr))
        target.append([0,0,1,0])
        m+=1
        if(m >= n):
            break

np.savez_compressed("data.npz", X=np.array(data), Y=np.array(target))