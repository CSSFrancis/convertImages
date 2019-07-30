#!/usr/bin/python

import sys, getopt
import os
import glob
import re
import numpy as np
import hyperspy.api as hs
import argparse
import empyer as em
import struct


"""
This program creates hdf5 files from a series of .emi files.  Either by stacking the files or dealing with each one
individually.


OPTIONS:
-d = cwd :directory
-f = cwd :output filename
-b = 2 : beam size
-s : stack the emi files?
-t: thickness of the sample
-k: the reciprocal space pixel size
-c: the convergence angle of the probe

"""


def read_param_file(file):
    with open(file) as f:
        params = f.read()
        metadata = {}
        print(params)
        try:
            metadata['microscope'] = re.search('(?<=microscope:)\s?(\w+)', params).group(1)
        except AttributeError:
            print("no microscope given")
            metadata['microscope'] = None

        try:
            metadata['camera_length'] = float(re.search('(?<=beam_size:)\s?(\d?\.\d*|\d+)', params).group(1))
        except AttributeError:
            print("no camera_length given")
            metadata['camera_length'] = None

        try:
            metadata['convergence_angle'] = float(re.search('(?<=convergence_angle:)\s?(\d?\.\d*|\d+)', params).group(1))
        except AttributeError:
            print("no convergence given")
            metadata['convergence_angle'] = None

        try:
            metadata['probe_size'] = float(re.search('(?<=probe_size:)\s?(\d?\.\d*|\d+)', params).group(1))
        except AttributeError:
            print("no probe_resolution given")
            metadata['probe_resolution'] = None

        try:
            metadata['pixel_size'] = float(re.search('(?<=pixel_size:)\s?(\d?\.\d*|\d+)', params).group(1))
        except AttributeError:
            print("no pixel-Size given")
            metadata['pixel_size'] = 1

        try:
            metadata['probe_step'] = float(re.search('(?<=probe_step:)\s?(\d?\.\d*|\d+)', params).group(1))
        except AttributeError:
            print("no probe step given")
            metadata['probe_step'] = 1
        try:
            metadata['deposition_rate'] = float(re.search('(?<=deposition:)\s?(\d?\.\d*|\d+)', params).group(1))
        except AttributeError:
            print("no deposition rate")
            metadata['deposition_rate'] = None
        try:
            t = str(re.search('(?<=thickness_slope:)\s?([-]?\d?\.\d*|[-]?\d+)', params).group(1))
            metadata['thickness_slope'] = float(t)
        except AttributeError:
            print("no thickness given")
            metadata['thickness_slope'] = None
        try:
            t = str(re.search('(?<=thickness_intercept:)\s?([-]?\d?\.\d*|[-]?\d+)', params).group(1))
            metadata['thickness_intercept'] = float(t)
        except AttributeError:
            print("no thickness_intercept given")
            metadata['thickness_intercept'] = None
        try:
            bin_file = str(re.search('(?<=bin_folder:)\s?(.+)', params).group(1))
            metadata['bin_folder'] = bin_file
        except AttributeError:
            print("no bin_folder given")
            metadata['bin_folder'] = None
    return metadata


def read_bin_files(folder):
    print(folder)
    files = os.listdir(folder)
    files = sorted(files)
    print(files)
    bin =[]
    for file in files:
        with open(folder+file, "rb") as f:
            binary = f.read()
        bin.append(np.reshape([struct.unpack_from('f', binary, i) for i in range(10, 407, 4)], (10, 10)))
    return bin


def repackage(input_directory,
              metadata,
              out_put_filename='out',
              stack=True):
    emi_files = sorted(glob.glob(input_directory+"/*.emi"))
    print(emi_files)
    names = [os.path.basename(f).split("emi")[0] for f in emi_files]
    if stack is True:
        s = hs.load(emi_files, stack=stack, new_axis_name="sample_position")

        s.axes_manager.signal_axes[0].scale = metadata['pixel_size']
        s.axes_manager.signal_axes[1].scale = metadata['pixel_size']
        s.axes_manager.signal_axes[0].name = "kx"
        s.axes_manager.signal_axes[1].name = "ky"
        s.axes_manager.signal_axes[0].units = "nm^-1"
        s.axes_manager.signal_axes[1].name = "nm^-1"
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        for axes in s.axes_manager.navigation_axes:
            axes.offset = 0
        s.metadata.Acquisition_instrument.TEM.probe_area = ((metadata['probe_size']/2)**2*np.pi)
        s.metadata.Acquisition_instrument.TEM.probe_resolution = metadata['probe_size']
        s.metadata.Acquisition_instrument.TEM.probe_step = metadata['probe_step']
        if not s.metadata.has_item("Sample"):
            s.metadata.add_node("Sample")
        s.metadata.Sample.thickness = metadata['thickness']
        s.metadata.Sample.deposition_rate = metadata['deposition_rate']
        s.metadata.original_filename = names
        s.metadata.convergence_angle = metadata['convergence_angle']
        s.metadata.Signal.signal_type = "diffraction_signal"
        s.save(input_directory + "/" + out_put_filename, extension='hdf5', overwrite=True)

    else:
        s = hs.load(emi_files)
        if metadata['bin_folder']:
            haadf_data = read_bin_files(metadata['bin_folder'])
        for i,(signal, name) in enumerate(zip(s, names)):
            signal = em.to_diffraction_signal(signal)
            signal.axes_manager.signal_axes[0].scale = metadata['pixel_size']
            signal.axes_manager.signal_axes[1].scale = metadata['pixel_size']
            signal.axes_manager.signal_axes[0].name = "kx"
            signal.axes_manager.signal_axes[1].name = "ky"
            signal.axes_manager.signal_axes[0].units = "nm^-1"
            signal.axes_manager.signal_axes[1].name = "nm^-1"
            signal.axes_manager.signal_axes[0].offset = 0
            signal.axes_manager.signal_axes[1].offset = 0
            for axes in signal.axes_manager.navigation_axes:
                axes.offset = 0
            signal.metadata.Acquisition_instrument.TEM.probe_area = ((metadata['probe_size'] / 2) ** 2 * np.pi)
            signal.metadata.Acquisition_instrument.TEM.probe_resolution = metadata['probe_size']
            signal.metadata.Acquisition_instrument.TEM.probe_step = metadata['probe_step']
            if not signal.metadata.has_item("Sample"):
                signal.metadata.add_node("Sample")
            signal.metadata.Sample.deposition_rate = metadata['deposition_rate']
            signal.metadata.original_filename = name
            signal.metadata.convergence_angle = metadata['convergence_angle']
            if metadata['bin_folder']:
                signal.add_hdaaf_intensities(haadf_data[i],
                                             slope=metadata['thickness_slope'],
                                             intercept=metadata['thickness_intercept'])
            print(signal.metadata)
            signal.save(input_directory + "/" + name, extension='hdf5', overwrite=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--directory",
                        type=str,
                        help="input directory")
    parser.add_argument("-f",
                        "--filename",
                        type=str,
                        help="output filename")
    parser.add_argument("-s",
                        "--stack",
                        help="Stack along a new axis",
                        action="store_true")
    parser.add_argument("-p",
                        "--paramfile",
                        type=str,
                        help="A parameter file to be read in")
    args = parser.parse_args()

    if not args.directory:
        input_directory = os.getcwd()
    else:
        input_directory = args.directory
    if not args.filename:
        output_filename = "output"
    else:
        output_filename = args.filename
    if args.stack:
        stack = True
    else:
        stack = False
    if not args.paramfile:
        print("You must have a parameter file!")
        exit()
    else:
        params = args.paramfile
    md = read_param_file(params)

    repackage(input_directory=input_directory,
              metadata=md,
              out_put_filename=output_filename,
              stack=stack)

