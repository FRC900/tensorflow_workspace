import numpy as np
import sys

def read_n_shuffled_lines(filename, yval, n):
    data_lines = np.genfromtxt(filename, delimiter=',', skip_header=1)
    np.random.shuffle(data_lines)
    data_lines = data_lines[:n]
    target = np.tile(yval, (n,1))

    return data_lines, target

n = 615 # Length of smallest input file so classes are balanced

# One-hot output encodings for each color
red_yval    = [1,0,0,0]
green_yval  = [0,1,0,0]
yellow_yval = [0,0,1,0]
blue_yval   = [0,0,0,1]

(red_xdata, red_ydata) = read_n_shuffled_lines('compiled_red.csv', red_yval, n)
(green_xdata, green_ydata) = read_n_shuffled_lines('compiled_green.csv', green_yval, n)
(yellow_xdata, yellow_ydata) = read_n_shuffled_lines('compiled_yellow.csv', yellow_yval, n)
(blue_xdata, blue_ydata) = read_n_shuffled_lines('compiled_blue.csv', blue_yval, n)

xdata = np.concatenate((red_xdata, green_xdata, yellow_xdata, blue_xdata), axis=0)
ydata = np.concatenate((red_ydata, green_ydata, yellow_ydata, blue_ydata), axis=0)

#np.set_printoptions(threshold=sys.maxsize)
#print (xdata)
#print ('------------------')
#print (ydata)
np.savez_compressed("unnormalized_data.npz", X=xdata, Y=ydata)
