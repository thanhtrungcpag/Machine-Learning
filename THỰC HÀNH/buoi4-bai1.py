import numpy as np
import matplotlib.pyplot as plt

X = np.array([0, 0.5, 1.5,2.5, 3])
Y = np.array([1, 2.5, 3.5, 4, 5.5])

plt.axis([0, 5, 0, 8])
plt.plot(X, Y, "ro", color = "blue")

plt.xlable("Gia Tri thuoc tinh X")
plt.xlable("Gia tri thuoc tinh Y")

plt.show()

def LR1(X, Y, eta, lanlap, theta0, theta1):
    m = len(X)
    for i in range(0, m):
        print("Lan Lap :", i)
        for j in range(0, m):
            h = theta0 + theta1*X[j]
            theta0 = theta0 + eta*(Y[j] - h)*1
            print("theta0 = ", theta0)
            theta1 = theta1 + eta*(Y[j] - h)*X[j]
            print("theta1", theta1)
    return  [theta0, theta1]

LR1(X, Y, 0.1, 3, 0, 0.5)