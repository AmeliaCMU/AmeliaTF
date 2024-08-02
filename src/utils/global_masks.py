import numpy as np

from easydict import EasyDict

VERSION = 7
#---------------------------------------------------------------------------------------------------
#                                       TRAJECTORY MASKS
# --------------------------------------------------------------------------------------------------
if VERSION == 5:
    # Raw index order from pre-processed CSV files:     
    RAW_IDX = EasyDict({
        'Frame':    0, 
        'ID':       1, 
        'Altitude': 2, 
        'Speed':    3, 
        'Heading':  4, 
        'Lat':      5, 
        'Lon':      6, 
        'Range':    7, 
        'Bearing':  8, 
        'Type':     9, 
        'x':       10, 
        'y':       11
    })
    # Mask removes 'Frame', 'ID' and 'AgentType'
    # Altitude, Speed, Heading, Lat, Lon, Range, Bearing, Type, x, y  
    RAW_SEQ_MASK = [False, False, True, True, True, True, True, True, True, False, True, True]
    
else: # Version 7
    RAW_IDX = EasyDict({
        'Frame':    0, 
        'ID':       1, 
        'Altitude': 2, 
        'Speed':    3, 
        'Heading':  4, 
        'Lat':      5, 
        'Lon':      6, 
        'Range':    7, 
        'Bearing':  8, 
        'Type':     9, 
        'Interp':  10, 
        'x':       11, 
        'y':       12 
    })
    # Mask removes 'Frame', 'ID' and 'AgentType'
    # Altitude, Speed, Heading, Lat, Lon, Range, Bearing, Type, Interp, x, y  
    RAW_SEQ_MASK = [False, False, True, True, True, True, True, True, True, False, False, True, True]

# New index order after mask
RAW_SEQ_IDX = EasyDict({
    'Altitude': 0, 
    'Speed':    1, 
    'Heading':  2, 
    'Lat':      3, 
    'Lon':      4, 
    'Range':    5, 
    'Bearing':  6,  
    'x':        7, 
    'y':        8
})

# Swapped order to go...
# From: Altitude, Speed, Heading, Lat, Lon, Range, Bearing, x, y
#   To: Speed, Heading, Lat, Lon, Range, Bearing, x, y, z
# SEQ_ORDER = [1, 2, 3, 4, 5, 6, 7, 8, 0]
SEQ_ORDER = [
    RAW_SEQ_IDX.Speed, 
    RAW_SEQ_IDX.Heading, 
    RAW_SEQ_IDX.Lat, 
    RAW_SEQ_IDX.Lon, 
    RAW_SEQ_IDX.Range,
    RAW_SEQ_IDX.Bearing, 
    RAW_SEQ_IDX.x, 
    RAW_SEQ_IDX.y, 
    RAW_SEQ_IDX.Altitude
]
# Final index order after post-processing: 
#   Speed, Heading, Lat, Lon, Range, Bearing, Interp, x, y, z (Previously Altitude)
SEQ_IDX = EasyDict({
    'Speed':   0, 
    'Heading': 1, 
    'Lat':     2, 
    'Lon':     3, 
    'Range':   4, 
    'Bearing': 5, 
    'x':       6, 
    'y':       7, 
    'z':       8, 
})

DIM = len(RAW_SEQ_IDX.keys())

# Agent types
AGENT_TYPES = {'Aircraft': 0, 'Vehicle': 1, 'Unknown': 2}

# TODO: what's this?
MAP_IDX = [False, False, True, True, False, False, True, True, True, False]

# A bit overkill, but it's to avoid indexing errors. 
# Traj Masks
LL = np.zeros(shape=len(SEQ_ORDER)).astype(bool)
LL[SEQ_IDX.Lat] = LL[SEQ_IDX.Lon] = True

HLL = np.zeros(shape=len(SEQ_ORDER)).astype(bool)
HLL[SEQ_IDX.Lat] = HLL[SEQ_IDX.Lon] = HLL[SEQ_IDX.Heading] = True

HD = np.zeros(shape=len(SEQ_ORDER)).astype(bool)
HD[SEQ_IDX.Heading] = True

XY = np.zeros(shape=len(SEQ_ORDER)).astype(bool)
XY[SEQ_IDX.x] = XY[SEQ_IDX.y] = True

XYZ = np.zeros(shape=len(SEQ_ORDER)).astype(bool)
XYZ[SEQ_IDX.x] = XYZ[SEQ_IDX.y] = XYZ[SEQ_IDX.z] = True

# REL_XY  = [ True,  True, False, False]
# REL_XY  = [ True,  True, False]
# REL_XYZ = [ True,  True,  True, False]
# REL_HD  = [False, False, False,  True]

# REL_XY  = [ True,  True, False, False]
REL_XY  = [ True,  True, False]
REL_XYZ = [ True,  True,  True, False, False]
REL_HD  = [False, False, False,  True, False]

# AIRPORT_COLORMAP = {
#     "kbos": "lightcoral",     #F08080
#     "kdca": "orangered",      #FF4500
#     "kewr": "gold",           #FFD700
#     "kjfk": "limegreen",      #32CD32
#     "klax": "darkturquoise",  #00CED1
#     "kmdw": "dodgerblue",     #1E90FF
#     "kmsy": "mediumorchid",   #BA55D3
#     "ksea": "violet",         #EE82EE
#     "ksfo": "deeppink",       #FF1493
#     "panc": "crimson",        #DC143C
# }