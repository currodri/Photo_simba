#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07 August 2019

@author: Curro Rodriguez Montero, School of Physics and Astronomy,
            University of Edinburgh, JCMB, King's Buildings

This code is a case example in which the position of a given galaxy is tracked in colour space. It uses the results stored
in the pickle file coming from the mergerFinder and quenchingFinder algorithms, as well the photometry data from the closer
routine.

For questions about the code:
s1650043@ed.ac.uk
"""

# Import required libraries
import numpy as np 
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white")
import cPickle as pickle
import sys

# Get details from terminal

MODEL = sys.argv[1]     # e.g. m100n1024
REDSHIFT = sys.argv[2]  # e.g. 0.5
GALAXY = [sys.argv[3]]    # e.g. 987
if len(sys.argv) > 4:
    GALAXY = sys.argv[4:] # this is for the case in which we want the tracks for multiple galaxies

# Read data from pickle file
data_file = '/home/curro/quenchingSIMBA/code/SH_Project/mandq_results_%s.pkl' % (MODEL)
obj = open(data_file, 'rb')
d = pickle.load(obj)

selected_galaxies = d['galaxies'][GALAXY]
