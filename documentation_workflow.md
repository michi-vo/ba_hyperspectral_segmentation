Flashing Clara:
	Host: Ubuntu 20.04, Slave: Ubuntu 20.04, Manual Mode, Pre-Config, Ivan: niklas
	Put Clara in Reset mode not recovery

Known Bugs:
	Ngc container:
		error: XDG_RUNTIME_DIR not set in the environment.
		--> Check on host: echo $DISPLAY
			returns e.g. :1
		--> set DISPLAY in container: export DISPLAY=:1 

Workflow developing in container (dgpu):
	1. docker login nvcr.io
	2. docker run -it --rm --net host \
  --runtime=nvidia \
  --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
  nvcr.io/nvidia/clara-holoscan/holoscan:v2.5.0-dgpu 

Run distributed application (e.g. video_replayer_distributed example):
	In 1. docker session:
		python3 video_replayer_distributed.py --driver --worker --address :1000 --fragments fragment1
	In 2. docker session (docker exec -it gifted_cray /bin/bash)
		python3 video_replayer_distributed.py --worker --address :1000 --fragments fragment2
	--> specified only port :1000 using guide IP: 192.168.50.68 didnt work
	
Workflow developing in holohub vscode devcontainer:
	1. [host]: start vscode: code --no-sandbox
	2. cd ~/holohub
	3. ./dev_container vscode

TO DO:
	docker container error Mellanox driver wrong? --> performance relevant?!
