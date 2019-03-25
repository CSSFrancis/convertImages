#!/usr/bin/python

import os
import glob
import re
import csv
import numpy as np
import argparse

# This program takes the prismatic outputs and then does a fft shift to reshape the matrix as well as
# it changes the format into a hyperspy output hdf5

"""
OPTIONS:
-d = cwd :directory
-f = cwd :output filename
-p : prismatic param file
-bf : binning factor

"""


def get_dsc_files(input_directory):
    dsc_files = glob.glob(input_directory + "/*.txt")  # get all text files

    for i, file in enumerate(dsc_files):
        with open(file, 'rt', encoding='ISO-8859-15') as dsc:
            [dsc.readline() for row in range(10)]
            for line in dsc.readlines():
                print(line)






if __name__ == '__main__':
    get_dsc_files('/home/carter/Documents/DSC' )
'''    parser = argparse.ArgumentParser()
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
'''
