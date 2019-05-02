from unittest import TestCase
import hyperspy.api as hs
from convert import get_mrc_files,repackage,convolve, gaussKernel, beam_convolution, real_fft_shift
import numpy as np
import matplotlib.pyplot as plt


class TestPolarSignal(TestCase):
    def setUp(self):
        x = np.random.rand(11, 11, 512, 512)
        pow_x = np.random.rand(512, 100,5, 5)
        self.signal = hs.signals.Signal2D(x, lazy=True)
        self.signal_2 = hs.signals.Signal2D(pow_x)
        print(self.signal.axes_manager.shape)

    def test_beamconvolution(self):
        c = beam_convolution(signal=self.signal, pixel_size=.05, beamsize=10)
        c.plot()
        self.signal.map(real_fft_shift, inplace=True, ragged=False)
        self.signal.isig[128:384, 128:384].plot()
        plt.show()

