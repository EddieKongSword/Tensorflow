import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

data=np.arange(1,16)
end_price=np.array([2511.90,2538.26,2510.68,2591.66,2732.98,2701.69,
                    2701.29,2678.67,2726.50,2681.50,2739.17,2715.07,2823.58,2864.90,2919.08])
begin_price = np.array([2438.71,2500.88,2534.95,2512.52,2594.04,2743.26,
                       2697.47,2695.24,2678.23,2722.13,2674.93,2744.13,2717.46,2832.73,2877.40])
plt.figure()
for i in range(0,15):
    dataOne=np.zeros([2])
    dataOne[0]=data[i]
    dataOne[1]=data[i]
    priceOne=np.zeros([2])
    priceOne[0]=begin_price[i]
    priceOne[1]=end_price[i]
    if  priceOne[0]>priceOne[1]:
        plt.plot(dataOne, priceOne,'g',lw=15,alpha=0.8 )
    else:
        plt.plot(dataOne, priceOne, 'r', lw=15,alpha=0.8)

plt.show()

