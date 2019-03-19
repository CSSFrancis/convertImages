import hyperspy.api as hs
import matplotlib.pyplot as plt

s = hs.load('/Users/shaw/Shaw/data/out5by5alphamax_0.74/output.hdf5')
s.plot()
plt.show()