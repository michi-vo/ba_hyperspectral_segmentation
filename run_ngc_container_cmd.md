docker run -it --rm --net host \
  --runtime=nvidia \
  --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
  --user user \
  --privileged \
  nvcr.io/nvidia/clara-holoscan/holoscan:v2.5.0-dgpu 

or

export NGC_CONTAINER_IMAGE_PATH="nvcr.io/nvidia/clara-holoscan/holoscan:v2.5.0-dgpu"

docker run -it --rm --net host \
  --runtime=nvidia \
  --ipc=host --cap-add=CAP_SYS_PTRACE --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY \
  ${NGC_CONTAINER_IMAGE_PATH} 
