from unittest import TestCase
import hyperspy.api as hs
import matplotlib.pyplot as plt
from convert_tiff import read_param_file, get_tif_files
from convert import get_mrc_files,repackage,convolve, gaussKernel
import numpy as np
import time


class TestPolarSignal(TestCase):
    def setUp(self):
        x = np.random.rand((11, 11, 62, 62))
        pow_x = np.random.rand((11, 11, 64, 64))
        self.kernel_1 = gaussKernel(1,62)
        self.kernel_2 = gaussKernel(1,64)
        self.signal = hs.BaseSignal(x, lazy=True)
        self.signal_2 = hs.BaseSignal(pow_x)
        print(self.signal.axes_manager.shape)

    def test_non_power(self):
        self.signal.map(convolve, inplace=True, ragged=False, gaussian=self.kernel1, show_progressbar=True)

    def test_power(self):
        self.signal_2.map(convolve,inplace=True, ragged=False, gaussian=self.kernel1, show_progressbar=True)
