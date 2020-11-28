import numpy as np

ab = np.array([[5,12,56,123,214,98], [56,895,45,12,215,663]])
print(ab)

#3D array works

n =  np.array([[1,2 ,3],[23,56,98], [36,87,21]])
print(n)
print(n[::2])

n[:,1,:] = [[9,9]]
print(n)