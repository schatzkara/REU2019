import numpy as np

a = {'a':1,'b':2}
b = a.copy()
b['a'] += 1
print(a)
print(b)
print(b.keys(), b.values())
print(list(b.keys())[0])

c = [2,2]
d = [1,2]
print(np.array(c) * np.array(d))
