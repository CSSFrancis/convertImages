import hyperspy.api as hs
import matplotlib.pyplot as plt
from convert_tiff import read_param_file, get_tif_files
from convert import get_mrc_files,repackage

#f ='/media/carter/c0020a70-8557-49b3-97e0-d5c6877a63fa/home/17-19-05.444_Export/Image/Raw/params.txt'
#read_param_file(f)
#get_tif_files('/media/carter/c0020a70-8557-49b3-97e0-d5c6877a63fa/home/17-19-05.444_Export/Image/Raw/')

repackage(input_directory='/media/carter/c0020a70-8557-49b3-97e0-d5c6877a63fa/home/TEMSimulations/20by202nmprobe',
          parameter_file= '/home/carter/Documents/TEMModeling/Tests/Simulation4/prismatic_gui_params.txt')