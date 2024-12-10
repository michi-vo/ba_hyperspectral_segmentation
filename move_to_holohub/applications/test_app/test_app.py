import sys
sys.path.append("/workspace/volumes/m2/data/ba_hyperspectral_segmentation/src")
sys.path.append('../../')


import scipy
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pysptools
import pysptools.spectro
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from scipy.linalg import pinv
from optimisation_helicoid import read_molecules_creatis, read_molecules_cytochrome_cb
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import HolovizOp
from holoscan.conditions import CountCondition, PeriodicCondition

class loadDataOp(Operator):
    def __init__(self, *args, **kwargs):
        # condition = PeriodicCondition(self, 1000)
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        print("loading biopsy probe file")
        data = 1
        print("sending loaded data")
        op_output.emit(data, "out")

class processDataOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        data = op_input.receive("in")
        print("received data in process op")
        rgb_data = data

        print("finished processing, sending to viz")
        op_output.emit(rgb_data, "out")

class HyperspectralVizOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("rgb")

    def compute(self, op_input, op_output, context):
        rgb = op_input.receive("rgb")
        print(rgb)


class biopsyApp(Application):
    def compose(self):
        loadData = loadDataOp(self, CountCondition(self, 1), name="loadData")
        processData = processDataOp(self, name="processData")
        viz = HyperspectralVizOp(self, name="viz")

        self.add_flow(loadData,processData)
        self.add_flow(processData, viz, {("out", "rgb")})
         

def main():
    app = biopsyApp()
    app.run()


if __name__ == "__main__":
    main()