import os
import shutil

from datetime import datetime

IN_DIR_V6 = './amelia_v6/debug_ksea'
IN_DIR_V5 = './amelia_v5/raw_trajectories/ksea'
OUT_DIR_V5 = './amelia_v5/debug_ksea'

ux_v5_timestamps = []

for f in os.listdir(IN_DIR_V6):
    # Get timestamp in V6
    FILE = os.path.join(IN_DIR_V6, f)
    ux_ts = int(FILE.split('/')[-1].split('_')[-1].split('.')[0])
    ts = datetime.utcfromtimestamp(ux_ts).strftime('%Y-%m-%d %H:%M:%S')
    

    # Check if timestamp exists in V5
    for f_v5 in os.listdir(IN_DIR_V5):
        if str(ux_ts) in f_v5:
            print(f"Found: Unix time: {ux_ts}, UTC time: {ts}")
            in_file = os.path.join(IN_DIR_V6, f)
            out_file = os.path.join(OUT_DIR_V5, f)
            shutil.copyfile(in_file, out_file)
            print(f"Copied file: {in_file} to {out_file}")
            break


