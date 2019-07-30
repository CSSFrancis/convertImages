from unittest import TestCase
import hyperspy.api as hs
import matplotlib.pyplot as plt
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


class Test_read_emi(TestCase):
    def setUp(self):
        self.input_file = '/media/hdd/home/HighTDatasets/Zr65Cu27.5Al7.5_1.19nmpsec(300W_3.8mT_170C_13sec)/1.19nmParameters.txt'

    def test_read(self):
        p = read_param_file(self.input_file)
        print(p)