#!/usr/bin/python

import sys, getopt
import os
import glob
import re
import numpy as np
import hyperspy.api as hs
import argparse


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


def repackage(input_directory,
              out_put_filename='out',
              stack=True,
              beamsize=2,
              thickness=None,
              scale=1,
              convergence_ang=None):
    emi_files = sorted(glob.glob(input_directory+"/*.emi"), key=os.path.getmtime)
    names = [os.path.basename(f).split("emi")[0] for f in emi_files]
    if stack is True:
        s = hs.load(emi_files, stack=stack, new_axis_name="sample_position")

        s.axes_manager.signal_axes[0].scale = scale
        s.axes_manager.signal_axes[1].scale = scale
        s.axes_manager.signal_axes[0].name = "kx"
        s.axes_manager.signal_axes[1].name = "ky"
        s.axes_manager.signal_axes[0].units = "nm^-1"
        s.axes_manager.signal_axes[1].name = "nm^-1"
        s.axes_manager.signal_axes[0].offset = 0
        s.axes_manager.signal_axes[1].offset = 0
        for axes in s.axes_manager.navigation_axes:
            axes.offset = 0
        s.metadata.Acquisition_instrument.TEM.probe_area = ((beamsize/2)**2*np.pi)
        s.metadata.Acquisition_instrument.TEM.probe_resolution = beamsize
        if not s.metadata.has_item("Sample"):
            s.metadata.add_node("Sample")
        s.metadata.Sample.thickness = thickness
        s.metadata.original_filename = names
        s.metadata.convergence_angle = convergence_ang
        s.metadata.Signal.signal_type = "diffraction_signal"
        print(s)
        s.save(input_directory + "/" + out_put_filename, extension='hdf5', overwrite=True)

    else:
        s = hs.load(emi_files)
        for signal, name in zip(s,names):
            signal.axes_manager.signal_axes[0].scale = scale
            signal.axes_manager.signal_axes[1].scale = scale
            signal.axes_manager.signal_axes[0].name = "kx"
            signal.axes_manager.signal_axes[1].name = "ky"
            signal.axes_manager.signal_axes[0].units = "nm^-1"
            signal.axes_manager.signal_axes[1].name = "nm^-1"
            signal.axes_manager.signal_axes[0].offset = 0
            signal.axes_manager.signal_axes[1].offset = 0
            for axes in signal.axes_manager.navigation_axes:
                axes.offset = 0
            signal.metadata.Acquisition_instrument.TEM.probe_area = ((beamsize / 2) ** 2 * np.pi)
            signal.metadata.Acquisition_instrument.TEM.probe_resolution = beamsize
            if not signal.metadata.has_item("Sample"):
                signal.metadata.add_node("Sample")
            signal.metadata.Sample.thickness = thickness
            signal.metadata.original_filename = name
            signal.metadata.convergence_angle = convergence_ang
            signal.metadata.Signal.signal_type = "diffraction_signal"
            print(signal)
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
    parser.add_argument("-b",
                        "--beamsize",
                        type=float,
                        help="Add in the beamsize in angstroms")
    parser.add_argument("-s",
                        "--stack",
                        help="Stack along a new axis",
                        action="store_true")
    parser.add_argument("-t",
                        "--thickness",
                        help="the thicnkess of the sample")
    parser.add_argument("-k",
                        "--scale",
                        type=float,
                        help="the reciprocal space scale")
    parser.add_argument("-c",
                        "--convergence",
                        type=float,
                        help="Microscope convergence angle")
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
    if not args.thickness:
        thickness = None
    else:
        thickness = args.thickness
    if not args.beamsize:
            beamsize = None
    else:
            beamsize = args.beamsize
    if not args.scale:
        scale = None
    else:
        scale = args.scale
    if not args.convergence:
        convergence_angle = None
    else:
        convergence_angle = args.convergence

    repackage(input_directory=input_directory,
              out_put_filename=output_filename,
              stack=stack,
              beamsize=beamsize,
              thickness=thickness,
              scale=scale,
              convergence_ang=convergence_angle)
