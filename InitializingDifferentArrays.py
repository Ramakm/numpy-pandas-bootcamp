#All 0s Matrix

import numpy as np
a = np.zeros((2,3))
print(a)
a = np.zeros((3,3,3))
print(a)

#All 1s Matrix

a = np.ones((4,2,2), dtype='int32')
print(a)

#All other values

a = np.full((2,2), 50)
print(a)

#All other values(Full like method)

a = np.full_like(a,4)
print(a)