import pybullet as p
import pybullet_data as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('infile', type=str)
# parser.add_argument('outfile', type='str', default=None, required=False)
args = parser.parse_args()

p.connect(p.DIRECT)
# name_in = os.path.join(pd.getDataPath(), "duck.obj")

name_out = os.path.splitext(args.infile)[0] + "_vhacd2.obj"  # if args.outfile is None else args.outfile
name_log = "log.txt"
p.vhacd(args.infile, name_out, name_log, alpha=0.04, resolution=1000000, concavity=0, maxNumVerticesPerCH=256)
