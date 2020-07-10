import numpy as np

markers_HSV = np.load("markers_HSV.npy")
print(markers_HSV)
print("----------------------------")
#print(markers_HSV[1][0][1])
#print(markers_HSV.shape)
temp = markers_HSV.reshape((3, 2, 3))
print("--------------------------\n", temp)
print(temp.shape)
print(temp)
print("----------------------")
print(temp[0])