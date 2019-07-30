#!/usr/bin/python

import os
import glob
import re
import numpy as np
import hyperspy.api as hs
import argparse

from PIL import Image

# This program takes the prismatic outputs and then does a fft shift to reshape the matrix as well as
# it changes the format into a hyperspy output hdf5

"""
OPTIONS:
-d = cwd :directory
-f = cwd :output filename
-p : prismatic param file
-bf : binning factor

"""


def read_param_file(file):
    with open(file) as f:
        params = f.read()
        metadata ={}
        try:
            metadata['microscope'] = re.search('(?<=microscope:)(\s*?\w+)', params).group(0)
        except(AttributeError):
            print("no microscope given")
            metadata['microscope'] = None

        try:
            metadata['acquisition_mode'] = re.search('(?<=acquisition_mode:)(\s*?\w+)', params).group(0)
        except(AttributeError):
            print("no acquisition mode given")
            metadata['acquisition_mode'] = None

        try:
            metadata['camera_length'] = float(re.search('(?<=camera_length:)(\s?\d?\.\d*|\s?\d+)', params).group(0))
        except(AttributeError):
            print("no camera_length given")
            metadata['camera_length'] = None

        try:
            metadata['convergence_angle'] = float(re.search('(?<=convergence_angle:)(\s?\d?\.\d*|\s?\d+)', params).group(0))
        except(AttributeError):
            print("no convergence given")
            metadata['convergence_angle'] = None

        try:
            metadata['beam_energy'] = float(re.search('(?<=beam_energy:)(\s?\d?\.\d*|\s?\d+)', params).group(0))
        except(AttributeError):
            print("no beam_energy given")
            metadata['beam_energy'] = None

        try:
            metadata['probe_resolution'] = float(re.search('(?<=probe_resolution:)(\s?\d?\.\d*|\s?\d+)', params).group(0))
        except(AttributeError):
            print("no probe_resolution given")
            metadata['probe_resolution'] = None

        try:
            metadata['dwell_time'] = float(re.search('(?<=dwell_time:)(\s?\d?\.\d*|\s?\d+)', params).group(0))
        except(AttributeError):
            print("no dwell_time given")
            metadata['dwell_time'] = None

        try:
            metadata['pixel_size'] = float(re.search('(?<=pixel_size:)(\s?\d?\.\d*|\s?\d+)', params).group(0))
        except(AttributeError):
            print("no pixel-Size given")
            metadata['pixel_size'] = 1

        try:
            metadata['probe_step'] = float(re.search('(?<=probe_step:)(\s?\d?\.\d*|\s?\d+)', params).group(0))
        except(AttributeError):
            print("no probe step given")
            metadata['probe_step'] = 1
        try:
            metadata['rows'] = int(re.search('(?<=rows:)(\s?\d?\.\d*|\s?\d+)', params).group(0))
        except(AttributeError):
            print("no rows given")
            metadata['rows'] = None
        try:
            metadata['columns'] = int(re.search('(?<=columns:)(\s?\d?\.\d*|\s?\d+)', params).group(0))
        except(AttributeError):
            print("no columns given")
            metadata['columns'] = None

    return metadata


def get_tif_files(input_directory, rows=None, columns=None, binning_factor=1):
    tif_files = glob.glob(input_directory + "/*.tif")  # get all tif files
    tiff_files = glob.glob(input_directory + "/*.tiff")  # get all tiff files
    print(tif_files)
    tif_files = tif_files+tiff_files
    print(tiff_files)
    # get positions
    tif_ind = np.array([re.search('(?<=_)(\d+)', f).group(1) for f in tif_files], dtype=int)
    print(tif_ind)
    # for indexing the files based on the output

    if rows ==None:
        #assume square
        max_ind = max(tif_ind) + 1
        rows = int(np.sqrt(max_ind))
        columns =rows

    index = sorted(range(len(tif_ind)), key=lambda k: tif_ind[k])
    # getting the size of the tif files
    with Image.open(tif_files[0]) as first:
        image_size = np.shape(bin_2d(np.array(first),binning_factor))
    image_data = np.zeros(((len(tif_files),) + image_size), dtype=np.float32)  # preallocate space
    for i, file in enumerate(tif_files):
        with Image.open(file) as tif:  # permissive as format is incorrect
            image_data[i] = bin_2d(np.array(tif), binning_factor)
    # shifting zero frequency component to the center

    # creating reformatting everything
    image_data = image_data[index]
    image_data = np.reshape(image_data, (rows, columns)+image_size)
    # averaging over phonon frequencies
    return image_data


def repackage(input_directory, output_filename, parameter_file, binning_factor=1):
    metadata = read_param_file(parameter_file)
    image_data = get_tif_files(input_directory, metadata['rows'], metadata['columns'], binning_factor=binning_factor)
    data_shape = np.shape(image_data)


    dict0 = {'size': data_shape[0], 'name': 'x', 'units': 'Angstroms', 'scale': metadata['probe_step'], 'offset': 0}
    dict1 = {'size': data_shape[1], 'name': 'y', 'units': 'Angstroms', 'scale': metadata['probe_step'], 'offset': 0}
    dict2 = {'size': data_shape[2], 'name': 'kx', 'units': 'nm^-1', 'scale': metadata['pixel_size']*binning_factor*10, 'offset': 0}
    dict3 = {'size': data_shape[3], 'name': 'ky', 'units': 'nm^-1', 'scale': metadata['pixel_size']*binning_factor*10, 'offset': 0}
    s = hs.signals.Signal2D(data=image_data, axes=[dict0, dict1, dict2, dict3])
    s.metadata.set_item("Acquisition_instrument.TEM.probe_resolution", metadata['probe_resolution'])
    s.metadata.set_item("Acquisition_instrument.TEM.acquisition_mode", metadata['acquisition_mode'])
    s.metadata.set_item("Acquisition_instrument.TEM.beam_energy", metadata['beam_energy'])
    s.metadata.set_item("Acquisition_instrument.TEM.camera_length", metadata['camera_length'])
    s.metadata.set_item("Acquisition_instrument.TEM.convergence_angle", metadata['convergence_angle'])
    s.metadata.set_item("Acquisition_instrument.TEM.microscope", metadata['microscope'])
    s.save(input_directory+"/"+output_filename, extension='hdf5', overwrite=True)

def bin_2d(image, binning_factor):
    """
    -This function takes a 2d image and then linearly interpolates to find the intermediate value.
    :param image:
    :param binning_factor:
    :return:
    """
    sh = np.shape(image)
    new_image = image.reshape(sh[0] // binning_factor, binning_factor, sh[1] // binning_factor, binning_factor).mean(
        -1).mean(1)

    return new_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=str,
                        help="input directory")
    parser.add_argument("-f", "--filename", type=str,
                        help="output filename")
    parser.add_argument("-p", "--parameters", type=str,
                        help="param file")
    parser.add_argument("-bf", "--binning", type=int,
                        help="binning factor for the image")
    args = parser.parse_args()
    if not args.directory:
        input_directory = os.getcwd()
    else:
        input_directory = args.directory

    if not args.filename:
        output_filename = "output"
    else:
        output_filename = args.filename

    if not args.parameters:
        print("You must have a parameter file ")
    else:
        params = args.parameters
    if not args.binning:
        bining =1
    else:
        binning= args.binning

    repackage(input_directory=input_directory,
              output_filename=output_filename,
              parameter_file=params,
              binning_factor=binning)

