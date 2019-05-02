#!/usr/bin/python

import sys, getopt
import os
import glob
import re
import numpy as np
import hyperspy.api as hs
import argparse

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
    index = sorted(range(len(mrc_ind)), key=lambda k: mrc_ind[k])
    mrc_files = mrc_files[index]  # ordering the mrc files based on FP, Y and X
    mrc_files = mrc_files.reshape(max_fp, max_y,max_x).tolist()
    stacked_signals = []
    for i, fp in enumerate(mrc_files):
        y_stack = []
        for y in fp:
            y_stack.append(hs.load(filenames=y, stack=True, new_axis_name='Real_x', lazy=True))
        hs.stack(y_stack, new_axis_name='Real_y').save(input_directory+"/fp"+str(i)+".hdf5", overwrite=True)
    phonons = glob.glob(input_directory + "/fp*.hdf5")
    signal = hs.load(phonons, stack=True, new_axis_name='Frozen_phonons', lazy =True)
    signal = signal.squeeze().mean(axis=2)
    signal.save(input_directory+"/cube.hdf5", overwrite=True)
    signal = hs.load(input_directory+"/cube.hdf5",lazy=True)
    return signal


def real_fft_shift(array):
    a = np.fft.fftshift(array).real
    return a


def beam_convolution(signal, pixel_size, beamsize):
    signal.map(real_fft_shift, inplace=True, ragged=False)
    shape = signal.axes_manager.signal_shape[0]//4
    signal = signal.isig[shape:-shape, shape:-shape]
    signal = signal.transpose(optimize=True)
    sigma = ((beamsize/pixel_size)/2.355)
    real_space = signal.axes_manager.signal_axes[0].size
    gk =gaussKernel(3, 10)
    plt.imshow(gk)
    plt.show()
    fkernel = np.fft.fft2(gaussKernel(sigma, real_space))
    signal.map(convolve, inplace=True, ragged=False, gaussian=fkernel, show_progressbar=False)
    signal = signal.transpose(optimize=True)
    return signal


def convolve(array, gaussian):
    a = np.fft.ifft2(gaussian * np.fft.fft2(array)).real
    return a


def gaussKernel(sigma, kernel_length):
    """
    creates a gaussian Kernal to be convolved
    :param sigma: the distrabution
    :param imsize: the size of the image
    :return: return a gaussian kernel based on the imsize and
    """
    ax = np.arange(-kernel_length // 2 + 1., kernel_length// 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))

    return kernel/np.sum(kernel)


def repackage(input_directory, parameter_file, out_put_filename='out', beamsize=2, convolve=True):
    x_pix, y_pix, cell_dim, tile_values, real_scale_x, real_scale_y = read_param_file(parameter_file)
    data = get_mrc_files(input_directory, average=True, shift=False)
    print(data)
    # number of pixels is equal to "real-space dimensions"\pixel-size
    # so the scale in reciprocal space is 1/dimensions
    kx_scale = 1 / ((cell_dim[0] / 10) * tile_values[0])
    ky_scale = 1 / ((cell_dim[1] / 10) * tile_values[1])
    if convolve:
        result = beam_convolution(data, pixel_size=real_scale_x, beamsize=beamsize)
    else:
        result = np.fft.fftshift(data).real

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
        input_directory = args.directory
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
