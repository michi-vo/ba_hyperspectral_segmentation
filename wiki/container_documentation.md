# I. Flashing Clara

Host: Ubuntu 20.04, Slave: Ubuntu 20.04, Manual Mode, Pre-Config, Ivan: niklas
Put Clara in Reset mode not recovery
regular install

* Known Bugs:
	* Ngc container:
		error: XDG_RUNTIME_DIR not set in the environment.
		--> Check on host: echo $DISPLAY
			returns e.g. :1
		--> set DISPLAY in container: export DISPLAY=:1 

# II. Creating a docker container environment

## II.1 Option 1 - Create a custom docker container (without default Dockerfile)
```bash
docker login nvcr.io

docker run -it --rm --net host \
  --runtime=nvidia \
  --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
  nvcr.io/nvidia/clara-holoscan/holoscan:v2.7.0-dgpu 
```
or
```bash
docker run -it --rm --net host \
  --runtime=nvidia \
  --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
  --user user \
  --privileged \
  nvcr.io/nvidia/clara-holoscan/holoscan:v2.7.0-dgpu 
```
* **update to latest holoscan:v2.x.x-dgpu or holohub:v2.x.x-dgpu**

### II.1.1 Run distributed application (e.g. video_replayer_distributed example):
In 1. docker session:
```bash
python3 video_replayer_distributed.py --driver --worker --address :1000 --fragments fragment1
```
In 2. docker session (docker exec -it gifted_cray /bin/bash)
```bash
python3 video_replayer_distributed.py --worker --address :1000 --fragments fragment2
```
--> specified only port :1000 using guide IP: 192.168.50.68 didnt work
	
## II.1.2 Option 2 - Developing in holohub vscode devcontainer
on host:
```bash
start vscode: code --no-sandbox
cd ~/holohub
./dev_container vscode
```
---> **Problem:** No external media mounting possible! (biopsy data)
---> I think it kept crashing after host reboot

## II.1.3 Option 3 - Use another container with additional parameters (CURRENTLY DEFAULT)
* Keep all files src and data in the /media/m2/files directory (keep data even if container crashes)
* https://github.com/nvidia-holoscan/holohub/tree/main/.devcontainer
* see holohub/dev_container (line 600)
```bash
	./dev_container launch --persistent --ssh_x11 --add-volume /media/m2
	exit
	docker container rename <container_name> hyperprobe_dev_container
	docker start hyperprobe_dev_container
	docker exec -it --user root hyperprobe_dev_container /bin/bash
	apt install libopenblas-dev
	apt-get install python3-tk
	apt install xdot
	exit
	docker exec -it hyperprobe_dev_container /bin/bash
	cd /workspace/volumes/media/m2/data/src
	pip install -r requirements.txt
```
* specify custom/latest holohub image: ./dev_container launch --persistent --ssh_x11 --img holohub:ngc-v2.7.0-dgpu

* Connect vscode to container:
	1. Contr+Shift+P
	2. Dev containers: Attach to running container

* Both the host and the container can modify the pulled repo but currently only the host can commit/push the changes

* Known bugs:
	* Cant connect to dev container inside vscode (remote)
		* In container:
		```bash
		rm -rf /workspace/holohub/.vscode-server
		```
		* close vscode and reopen container to download vscode-server files 
	* Container doesn't start after reboot of host machine because of "device not found: snd"
		* In holohub/dev_container: Comment out: 
		```bash
		# Find all audio devices
		# if [ -d /dev/snd ]; then
		#     audio_devices=$(find /dev/snd -type c)

		#     # Mount all found audio devices
		#     for audio_dev in $audio_devices; do
		#         mount_device_opt+="${prefix}--device=$audio_dev${postfix}"
		#     done
		# fi 
		```
	* Container keeps breaking after each reboot?!
	* Interactive image output not working (plt.imshow(rgb))
		```bash
		gi.repository.GLib.GError: gdk-pixbuf-error-quark: Couldn’t recognize the image file format for file “/workspace/holohub/.local/lib/python3.10/site-packages/matplotlib/mpl-data/images/matplotlib.svg” (3)
		```
		* Set TkAgg backend for interactive plots
		```python
		import matplotlib
		matplotlib.use('TkAgg')
		```
### II.1.3.1 Build ngc image
* Download latest holoscan container image (https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#prerequisites)
```bash
docker pull nvcr.io/nvidia/clara-holoscan/holoscan:v2.7.0-dgpu
```
* Pull latest version of holohub git
```bash
./dev_container build
```
### II.1.3.2 Future options
#### II.1.3.2.1 Export Container (if stable container version is achieved)
```bash
docker export ...
```
#### II.1.3.2.2 Install sudo
* install sudo to prevent having to log into the container as root each time

## III. Data
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

## IV. Benchmarking
### Run holoscan flow benchmarking
1. Log into container as root: 
```bash
apt install gir1.2-gtk-3.0 python3-gi python3-gi-cairo python3-numpy graphviz
```

## V. TO DO:
	docker container error Mellanox driver wrong? --> performance relevant?!

## VI. General resources: 
	* https://ieeexplore.ieee.org/document/9562441/keywords#keywords


## VII. Questions:
	* [biopsy.ipynb]: [12]: load t1 which is not existant, then in the last line save that/or similar t1 (first use then create?)
