# Changes in code
* In biopsy.ipynb: change the file path depending on your system configuration
--> use dynamic path allocation

## Benchmark application

### Build and run
1. Move app into holohub/applications
2. create metadata.json:
```json
{
	"application": {
		"name": "Biopsy Application",
		"authors": [
			{
				"name": "Holoscan Team",
				"affiliation": "NVIDIA"
			}
		],
		"language": "Python",
		"version": "1.0",
		"changelog": {
			"1.0": "Initial Release"
		},
		"holoscan_sdk": {
			"minimum_required_version": "0.5.0",
			"tested_versions": [
				"0.5.0"
			]
		},
		"platforms": [
			"amd64",
			"arm64"
		],
		"tags": [
			"Colonoscopy",
			"Classification"
		],
		"ranking": 1,
		"dependencies": {
			"data": [
				{
					"name": "Holoscan Sample App Data for AI Colonoscopy Segmentation of Polyps",
					"description": "This resource contains a segmentation model for the identification of polyps during colonoscopies trained on the Kvasir-SEG dataset [1], using the ColonSegNet model architecture [2], as well as a sample surgical video.",
					"url": "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_colonoscopy_sample_data"
				}
			]
		},
		"run": {
			"command": "python3 test_app.py",
			"workdir": "holohub_app_source"
		}
	}
}
```
3. In holohub/ 
```bash
./run setup
#for biopsy_app (currently broken)
./benchmarks/holoscan_flow_benchmarking/patch_application.sh applications/biopsy_app
./run build biopsy_app --benchmark 
# if app cannot be found revert git changes
python benchmarks/holoscan_flow_benchmarking/benchmark.py -a biopsy_app --language python -r 3 -i 3 --sched greedy -d myoutputs --level debug

#for test_app
./benchmarks/holoscan_flow_benchmarking/patch_application.sh applications/test_app
./run build test_app --benchmark
python benchmarks/holoscan_flow_benchmarking/benchmark.py -a test_app --language python -r 3 -i 3 --sched greedy -d myoutputstest --level debug

./run launch biopsy_app python
```
### Plot results
#### Show latencies
```bash
python benchmarks/holoscan_flow_benchmarking/analyze.py -m -a -g myoutputstest/logger_greedy_* MyCustomGroup
```
--> This display is only working if multiple runs have been performed (e.g. -r 3)
#### Draw cdf
```bash
python benchmarks/holoscan_flow_benchmarking/analyze.py --draw-cdf single_path_cdf.png -g myoutputstest/logger_greedy_* MyCustomGroup --no-display-graphs
```
#### Draw graph with individual latencies
```bash
python benchmarks/holoscan_flow_benchmarking/app_perf_graph.py myoutputstest/logger_greedy_1_1.log -o myoutputstest/test.dot
xdot myoutputstest/test.dot
```
#### Draw live graph
```bash
python benchmarks/holoscan_flow_benchmarking/benchmark.py -a test_app --language python -r 3 -i 3 --sched greedy -d running_live_graph --level debug
python3 benchmarks/holoscan_flow_benchmarking/app_perf_graph.py -o live_app_graph.dot -l running_live_graph
xdot live_app_graph.dot
```
--> open each one in separate terminal

### Restore application
```bash
./benchmarks/holoscan_flow_benchmarking/restore_application.sh test_app
```

### Useful commands

* Cleanup
You can run the command below to reset your build directory:
```bash
./run clear_cache
```
In some cases you may also want to clear out datasets downloaded by applications to the data folder:
```bash
rm -rf ./data
```

### Bugs and Problems

* With loading operator:
```python
```
* Wrong permissions:
```bash
	chown -R ivan:ivan /workspace/
	chown -R ivan:ivan /var/lib/apt/lists
	chown -R ivan:ivan /etc
	chown -R ivan:ivan /var/cache/apt/archives/partial/
```

### Possible improvements

* convert .mat files into blosc binary files (and float16 might be sufficient)
	Biopsy cube dimensions (example):

	biopsy_cube.shape  # (2048, 2048, 224) 

	2048x2048 spatial resolution

	224 spectral bands

	Float32 data type = 2048 * 2048 * 224 * 4 bytes ≈ 3.75GB

	filepath: /workspace/holohub/applications/hyperspectral_segmentation/hyperspectral_segmentation.py
	
	Segmentation cube dimensions (example):

	segmentation_cube.shape  # (512, 512, 128)

	512x512 spatial resolution

	128 spectral bands

	Float32 data type = 512 * 512 * 128 * 4 bytes ≈ 134MB

* Cut cube into smaller chunks (paralellisation)
* Use cuda/cudnn
* Use neural network (already available?) instead of least squares


### Least Squares

The Least Squares Problem:
The goal is to find the parameters (delta_c_i, a_ti, b_ti) that minimize the sum of squared residuals:
minimize Σ(residuals²)
where residuals = f(delta_c_i, a_ti, b_ti; b[i], b_t1, a_t1, M, x)
In other words, we are trying to find the concentration changes and scattering parameters that best explain the observed data, given the previous scattering parameters and the known absorption coefficients.
The non-linearity comes from the scattering term S(...), which makes this a non-linear least squares problem. The scipy.optimize.least_squares function uses the Levenberg-Marquardt algorithm or a trust region reflective algorithm to solve this non-linear optimization problem.
This optimization is performed for each pixel or time point i, allowing the concentrations and scattering parameters to vary across the image or time series.

### Additional resources

* Load .mat files as binary (C): 

https://stackoverflow.com/questions/32332920/efficiently-load-a-large-mat-into-memory-in-opencv/32357875#32357875

### cupy vs pytorch
Certainly! Profiling your code with both CuPy and PyTorch CUDA implementations is crucial for optimizing performance in your specific use case. Here's a more detailed explanation of why and how to do this:

1. Different strengths:

CuPy is designed to be a drop-in replacement for NumPy with GPU acceleration. It excels at array operations and scientific computing tasks.

PyTorch CUDA is optimized for deep learning operations and provides a dynamic computational graph, which can be beneficial for certain types of computations.

2. Operation-specific performance:

Some operations might be faster in CuPy, while others might be faster in PyTorch CUDA.

For example, basic array operations and certain linear algebra computations might be faster in CuPy, while convolutions and other neural network-specific operations might be faster in PyTorch.

3. Data transfer overhead:

Moving data between CPU and GPU can be a significant bottleneck.

CuPy and PyTorch might have different overheads for data transfer, which can affect overall performance.

4. Memory management:

CuPy and PyTorch have different memory management strategies, which can impact performance, especially for large datasets or memory-intensive operations.
To profile your code effectively:

1. Isolate operations:

Implement the same operation using both CuPy and PyTorch CUDA. For example:
```python
import cupy as cp
import torch
import time

# CuPy implementation
def cupy_operation(data):
    start_time = time.time()
    result = cp.some_operation(data)
    cp_time = time.time() - start_time
    return result, cp_time

# PyTorch CUDA implementation
def pytorch_operation(data):
    start_time = time.time()
    result = torch.some_operation(data.cuda())
    torch.cuda.synchronize()  # Ensure all CUDA operations are completed
    torch_time = time.time() - start_time
    return result.cpu(), torch_time

# Run and compare
data = np.random.rand(1000, 1000)
cupy_result, cupy_time = cupy_operation(data)
pytorch_result, pytorch_time = pytorch_operation(torch.from_numpy(data))

print(f"CuPy time: {cupy_time}")
print(f"PyTorch time: {pytorch_time}")
print(f"Results match: {np.allclose(cupy_result, pytorch_result)}")
```
2. Use profiling tools:

For CuPy: Use cupyx.profiler.benchmark or nvprof

For PyTorch: Use torch.autograd.profiler.profile or nvprof

3. Test with different data sizes:

Performance characteristics might change with different input sizes, so test with various data dimensions relevant to your use case.

4. Consider end-to-end pipeline:

While individual operations are important, also profile your entire processing pipeline to account for data movement and other overheads.

5. Iterate and optimize:

Based on profiling results, optimize your code by choosing the faster implementation for each operation or by restructuring your pipeline to minimize data transfers.

Here's an example of how you might profile a specific operation:
```python
import cupy as cp
import torch
import cupyx.profiler
from torch.autograd.profiler import profile

def cupy_operation(data):
    return cp.mean(data, axis=0)

def pytorch_operation(data):
    return torch.mean(data, dim=0)

# Prepare data
data = np.random.rand(1000, 1000)
cupy_data = cp.asarray(data)
torch_data = torch.from_numpy(data).cuda()

# Profile CuPy
with cupyx.profiler.time_range("cupy_mean"):
    cupy_result = cupy_operation(cupy_data)

# Profile PyTorch
with profile(use_cuda=True) as prof:
    pytorch_result = pytorch_operation(torch_data)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# Compare results
print(f"Results match: {np.allclose(cupy_result.get(), pytorch_result.cpu().numpy())}")
By systematically profiling and comparing the performance of CuPy and PyTorch CUDA for your specific operations and data, you can make informed decisions about which library to use for different parts of your application, ultimately optimizing the overall performance of your Holoscan application.
```
