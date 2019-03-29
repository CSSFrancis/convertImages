#!/usr/bin/python

import sys, getopt
import os
import glob
import re
import numpy as np
import hyperspy.api as hs
import argparse
import scipy
import dask
import matplotlib.pyplot as plt


# This program takes the prismatic outputs and then does a fft shift to reshape the matrix as well as
# it changes the format into a hyperspy output hdf5

"""
OPTIONS:
-d = cwd :directory
-f = cwd :output filename
-p : prismatic param file
-b = 2 : beamsize

"""


def read_param_file(file):
    with open(file) as f:
        params = f.read()
        print(params)
        x_pix= float(re.search('--pixel-size-x:([-+]?\d*\.\d+|\d+)', params).groups()[0])
        y_pix = float(re.search('--pixel-size-y:([-+]?\d*\.\d+|\d+)', params).groups()[0])
        tile_values = np.array(re.search(
                            '--tile-uc:([-+]?\d*\.\d+|\d+) ([-+]?\d*\.\d+|\d+) ([-+]?\d*\.\d+|\d+)', params).groups(1),
                            dtype=float)
        cell_dim = np.array(re.search(
            '--cell-dimension:([-+]?\d*\.\d+|\d+) ([-+]?\d*\.\d+|\d+) ([-+]?\d*\.\d+|\d+)', params).groups(1),
            dtype=float)
        real_scale_x = float(re.search('--probe-step-x:([-+]?\d*\.\d+|\d+)', params).groups()[0])
        real_scale_y = float(re.search('--probe-step-y:([-+]?\d*\.\d+|\d+)', params).groups()[0])
        print(cell_dim)
        print(tile_values)
    return x_pix, y_pix, cell_dim, tile_values, real_scale_x, real_scale_y


def get_mrc_files(input_directory, average=True, shift=False):
    mrc_files = np.array(glob.glob(input_directory + "/*_X*_Y*.mrc"))  # get only inverse space images
    # get positions and phonon number
    mrc_ind = np.array([[re.search('_X(\d+)', f).group(1),
                         re.search('_Y(\d+)', f).group(1),
                         int(re.search('_FP(\d+)', f).group(1))] for f in mrc_files], dtype=int)
    # for indexing the files based on the output
    max_x, max_y, max_fp = max(list(zip(*mrc_ind))[0])+1,max(list(zip(*mrc_ind))[1])+1, max(list(zip(*mrc_ind))[2])
    mrc_ind = [ind[0]+ind[1]*max_x+ind[2]*max_x*max_y for ind in mrc_ind]  # in order based on x,y and fp
    print(mrc_ind)
    index = sorted(range(len(mrc_ind)), key=lambda k: mrc_ind[k])
    print(index)
    mrc_files = mrc_files[index]  # ordering the mrc files based on FP, Y and X
    mrc_files = mrc_files.reshape(max_fp, max_y,max_x).tolist()
    stacked_signals = []
    for fp in mrc_files:
        y_stack = []
        for y in fp:
            y_stack.append(hs.load(filenames=y, stack=True, new_axis_name='Real_x', lazy=True))
        stacked_signals.append(hs.stack(y_stack, new_axis_name='Real_y'))
        print('Y')
    signal = hs.stack(stacked_signals, new_axis_name='frozen_phonon').squeeze().mean(axis=2)
    return signal


def beam_convolution(signal, pixel_size, beamsize):
    print(signal.data)
    signal = signal.transpose(optimize=True)
    sigma = ((beamsize/pixel_size)/2.355)
    real_space = signal.axes_manager.signal_axes[0].size
    print(real_space)
    fkernel = np.fft.fft2(gaussKernel(sigma, real_space))
    print('here')
    print(signal.axes_manager)
    print(signal.data)
    print(signal.data[1,1,:,:])
    convolve(signal.data[1,1,:,:], fkernel)
    signal.map(convolve, inplace=True, ragged=False, gaussian=fkernel)
    signal = signal.transpose(optimize=True)
    return signal


def convolve(array, gaussian):
    a = dask.array.fft.ifft2(gaussian * dask.array.fft.fft2(array))
    a = dask.array.fft.fftshift(a).real
    print(np.shape(a))


def gaussKernel(sigma, imsize):
    """
    creates a gaussian Kernal to be convolved
    :param sigma: the distrabution
    :param imsize: the size of the image
    :return: return a gaussian kernel based on the imsize and
    """
    x, y = np.meshgrid(range(1,imsize+1),range(1,imsize+1))
    x = x - imsize//2
    y = y - imsize//2
    tmp = -(x**2+y**2)/(2*sigma**2)
    return (1/(2*np.pi*sigma**2))*np.exp(tmp)


def repackage(input_directory, parameter_file, out_put_filename='out', beamsize=2, convolve=True):
    x_pix, y_pix, cell_dim, tile_values, real_scale_x, real_scale_y = read_param_file(parameter_file)
    data = get_mrc_files(input_directory, average=True, shift=False)
    print(data)
    # number of pixels is equal to "real-space dimensions"\pixel-size
    # so the scale in reciprocal space is 1/dimensions
    kx_scale = 1 / ((cell_dim[0] / 10) * tile_values[0])
    ky_scale = 1 / ((cell_dim[1] / 10) * tile_values[1])
    if convolve:
        result = beam_convolution(data, pixel_size=x_pix, beamsize=beamsize)
    else:
        result = np.fft.fftshift(data).real
        outside_mask_kx = data_shape[-2]//4
        outside_mask_ky = data_shape[-1] // 4
        result = result[:, :, outside_mask_kx:outside_mask_kx*3,outside_mask_ky:outside_mask_ky*3]
    dict0 = {'size': data_shape[0], 'name': 'x', 'units': 'Angstroms', 'scale': real_scale_x, 'offset': 0}
    dict1 = {'size': data_shape[1], 'name': 'y', 'units': 'Angstroms', 'scale': real_scale_y, 'offset': 0}
    dict2 = {'size': data_shape[2], 'name': 'kx', 'units': 'nm^-1', 'scale': kx_scale, 'offset': 0}
    dict3 = {'size': data_shape[3], 'name': 'ky', 'units': 'nm^-1', 'scale': ky_scale, 'offset': 0}
    s = hs.signals.Signal2D(data=result, axes=[dict0, dict1, dict2, dict3])
    s.save(input_directory+"/"+out_put_filename, extension='hdf5', overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str,
                        help="input directory")
    parser.add_argument("-f", "--filename", type=str,
                        help="output filename")
    parser.add_argument("-p", "--parameters", type=str,
                        help="prismatic parameter file")
    parser.add_argument("-b", "--beamsize",type=float,
                        help="Add in the beamsize in angstroms")
    parser.add_argument("-c", "--convolve", help="prismatic parameter file", action="store_true")
    args = parser.parse_args()
    if not args.directory:
        input_directory = os.getcwd()
    else:
        input_directory= args.directory

    if not args.filename:
        output_filename = "output"
    else:
        output_filename =args.filename

    if not args.convolve:
        convolve = True
    else:
        convolve = False

    if not args.parameters:
        print("You must have a parameter file ")
    else:
        params = args.parameters
    if not args.beamsize:
            beamsize = 20
    else:
            beamsize = args.beamsize

    repackage(input_directory=input_directory,
              out_put_filename=output_filename,
              parameter_file=params,
              beamsize=beamsize,
              convolve=convolve)

