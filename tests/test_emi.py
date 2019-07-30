from unittest import TestCase

from convert_emi import read_param_file, repackage


class Test_read_emi(TestCase):

    def setUp(self):
        self.input_file = '/home/carter/Documents/ZrCuAl(HH)03-28-2018_300W_3.8mT_170C_13s(1.19nmpsec)/1.19nmParameters.txt'

    def test_read(self):
        p = read_param_file(self.input_file)
        print(p)

    def test_repackage(self):
        p = read_param_file(self.input_file)
        direct = '/home/carter/Documents/ZrCuAl(HH)03-28-2018_300W_3.8mT_170C_13s(1.19nmpsec)/emi_ser_files'
        repackage(input_directory=direct, metadata=p, stack=False, out_put_filename="830pmpsec")
