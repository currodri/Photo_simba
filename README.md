# Photo_simba
Python version of the original code by Vivienne Wild for the SC analysis, adapted to SIMBA simulation data.

September 2019 - beta edition

Code description
================

Galaxy classification is a tricky business, and couldn't be different for the case of cosmological simulations.
In this repository you can find different Python codes that make use of the U, V and J colours from the closer (https://bitbucket.org/romeeld/closer/src/default/)
or pyloser (https://pyloser.readthedocs.io/en/latest/) catalogues,
In addition to this, we provide too codes that obtain and use the results of the Super Color classification, 
as especified in Wild et al. 2007, with the theroretical inspiration of Connolly et al. 1995.

1) The main routines of the normalized Super Color projection of a gappy spectrum are given in pysca.py. You need to be aware that,
    while the Super Color is a well tested classification method, it is strongly reliant on the filters used and the type of magnitudes.
    Make sure that, if the eigenbasis have been computed in apparent magnitudes with a given set of filters, that the sample of galaxies
    from the simulation presents the apparent magnitudes in the same filters.
2) The loser_extractor.py code cross-matches the results from the photometry files in Simba with the galaxy catalogues from the
    mergerFinder and quenchingFinder algorithms (for more information, visit https://github.com/Currodri/MQR_simba).
3) sc_plots.py and uvj_plots.py provide simple plotting codes that use the cross-matched catalogues of photometry, mergers and quenching
    to create single scatter plots.
4) trad_psb_select.py provides functions that obtain equivalwnt widths for particular lines in the pyloser spectra that can be used to
    classify galaxies as PSBs.
5) colour_tracking.py is an interesting example of the usefulness of the photometric analysis in multiple snapshots to test the evolution
    in colour and SC diagrams of a given galaxy.

Eigensystems
============

The eigensystem used for the testing of these codes was made by Viviene Wild using a super-sampled grid of broad-band fluxes, 
shifted by dz=0.01 for 0.5 < z < 3.0. The arrays are then chopped to ensure there are no rest-frame wavelengths < minwave=2500AA and > maxwave = 15000AA. If you would like to use this for your analysis, please contact me (s1650043@ed.ac.uk) or Vivienne Wild (vw8@st-andrews.ac.uk)
for permission.
