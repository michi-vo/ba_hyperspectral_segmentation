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
./run setup.py
#for biopsy_app (currently broken)
./benchmarks/holoscan_flow_benchmarking/patch_application.sh applications/biopsy_app
./run build biopsy_app --benchmark 
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

### Bugs and Problems

* With loading operator:
```python
