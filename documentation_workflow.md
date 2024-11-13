* Flashing Clara:
	Host: Ubuntu 20.04, Slave: Ubuntu 20.04, Manual Mode, Pre-Config, Ivan: niklas
	Put Clara in Reset mode not recovery

* Known Bugs:
	* Ngc container:
		error: XDG_RUNTIME_DIR not set in the environment.
		--> Check on host: echo $DISPLAY
			returns e.g. :1
		--> set DISPLAY in container: export DISPLAY=:1 

* Workflow developing in container (dgpu):
	1. docker login nvcr.io
	2. docker run -it --rm --net host \
  --runtime=nvidia \
  --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
  nvcr.io/nvidia/clara-holoscan/holoscan:v2.5.0-dgpu 

* Run distributed application (e.g. video_replayer_distributed example):
	In 1. docker session:
		python3 video_replayer_distributed.py --driver --worker --address :1000 --fragments fragment1
	In 2. docker session (docker exec -it gifted_cray /bin/bash)
		python3 video_replayer_distributed.py --worker --address :1000 --fragments fragment2
	--> specified only port :1000 using guide IP: 192.168.50.68 didnt work
	
* Workflow developing in holohub vscode devcontainer:
	1. [host]: start vscode: code --no-sandbox
	2. cd ~/holohub
	3. ./dev_container vscode
	---> **Problem:** No external media mounting possible! (biopsy data)

* Workflow for using another container with additional parameters
	* https://github.com/nvidia-holoscan/holohub/tree/main/.devcontainer
	* see holohub/dev_container (line 600)
	```bash
		./dev_container launch --persistent --ssh_x11
		exit
		docker container rename <container_name> hyperprobe_dev_container
		docker start hyperprobe_dev_container
		docker exec -it --user root hyperprobe_dev_container /bin/bash
		apt install libopenblas-dev
		exit
		docker exec -it hyperprobe_dev_container /bin/bash
		cd /workspace
	```
	* Connect vscode to container:
		1. Contr+Shift+P
		2. Dev containers: Attach to running container

	* Copy code e.g. into /workspace/src: 
	
		https://syncandshare.lrz.de/getlink/fiTjTyRrfJAZeyjmGMKsz1/michael.zip
	* Download biopsy data: 

		https://zenodo.org/records/10908359
	* Download UCL-NIR-Sepctra: 

		https://github.com/multimodalspectroscopy/UCL-NIR-Spectra/tree/main/spectra
	* Download HSI data?:

		https://hsibraindatabase.iuma.ulpgc.es/

	* additional resources: 

		https://github.com/HyperProbe
	


* TO DO:
	docker container error Mellanox driver wrong? --> performance relevant?!

* General resources: 
	* https://ieeexplore.ieee.org/document/9562441/keywords#keywords


* Questions:
	* [biopsy.ipynb]: [12]: load t1 which is not existant, then in the last line save that/or similar t1 (first use then create?)