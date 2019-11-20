import numpy as np



a = np.zeros((3,4))
a[0,:] = [1,2,3,4]
a[1,:] = [5,6,7,8]
a[2,:] = [9,10,11,12]

print(a)

print("-"*10)


print(np.diff(a))