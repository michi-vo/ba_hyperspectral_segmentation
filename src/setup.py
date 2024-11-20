import sys
sys.path.append("/workspace/holohub/benchmarks/holoscan_flow_benchmarking")
from benchmarked_application import BenchmarkedApplication

import os

os.mkdir("./results")
os.mkdir("./results/model_checkpoints")
os.mkdir("./dataset")
os.mkdir("./dataset/synthetic")
os.mkdir("./dataset/piglet_diffs")
os.system("git clone https://github.com/multimodalspectroscopy/UCL-NIR-Spectra.git ./dataset/UCL-NIR-Spectra")

os.system("pip install -r requirements.txt")


