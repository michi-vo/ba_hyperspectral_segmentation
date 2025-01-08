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