# Setting up Nvidia Clara

# I. Flashing Clara

Host: Ubuntu 20.04, Slave: Ubuntu 20.04, 

* Option 1: Automatic Mode, Pre-Config, Ivan: niklas (if reflashing)

* Option 2: Manual Mode, Pre-Config, Ivan: niklas

   * Put Clara in Reset mode not recovery
   * regular install

* Connect HDMI cable to Clara (Displayport only works when enabling dGPU later on)

* Known Bugs:
	* Ngc container:
		error: XDG_RUNTIME_DIR not set in the environment.
		--> Check on host: echo $DISPLAY
			returns e.g. :1
		--> set DISPLAY in container: export DISPLAY=:1 

## Switching between iGPU and dGPU
The Clara AGX Developer Kit can use either the Xavier AGX module GPU (iGPU, integrated GPU)
or the RTX6000 add-in card GPU (dGPU, discrete GPU). You can only use one type of GPU at a time.

By default, the Clara AGX Developer Kit uses the iGPU. Switching between the iGPU and dGPU
is performed using the *nvgpuswitch.py* script located in the */opt/nvidia/l4t-gputools/bin/directory*.
To make the *nvgpuswitch.py* script accessible globally, copy it to a directory included in *$PATH* if it hasn't been already:
```
$ sudo cp /opt/nvidia/l4t-gputools/bin/nvgpuswitch.py /usr/local/bin/
```

To switch from the iGPU to the dGPU, follow these steps:

1. Connect the Clara AGX Developer Kit to the Internet using one of the following methods:
    * An Ethernet cable connected to a router or Wi-Fi extender
        1. Use the 1GbE connector to the Xavier module at port 7.
    * A USB Wi-Fi receiver
        1. Not all USB Wi-Fi receivers will work out of the box on the Clara AGX Developer Kit.
        2. The USB Wi-Fi receiver should have support for Ubuntu 20.04.
        3. The TP-Link Archer T2U Nano USB Wi-Fi Adapter, which has previously worked with the Ubuntu 18.04 Holoscan SDK versions, will no longer work with Holoscan SDK v0.2, which based on Ubuntu 20.04.

2. To view the currently installed drivers and their version, use the _query_ command:
```
$ nvgpuswitch.py query
iGPU (nvidia-l4t-cuda, 34.1.2-20220613164700)
```

3. To install the dGPU drivers, use the _install_ command with the *dGPU* parameter (note that *sudo* must be used to install drivers):

```
$ sudo nvgpuswitch.py install dGPU
```
    
   The *install* command prints out the list of commands that will be executed as part of the driver install and then continues to execute those commands. This aids with debugging if any of the commands fail to execute.

   The following arguments may also be provided with the *install* command :
```    
$ nvgpuswitch.py install -h
usage: nvgpuswitch.py install [-h] [-f] [-d] [-i] [-v] [-l LOG] [-r [L4T_REPO]] {iGPU,dGPU}
 
positional arguments:
  {iGPU,dGPU}           install iGPU or dGPU driver stack
 
optional arguments:
  -h, --help            show this help message and exit
  -f, --force           force reinstallation of the specified driver stack
  -d, --dry             do a dry run, showing the commands that would be executed but not actually executing them
  -i, --interactive     run commands interactively (asks before running each command)
  -v, --verbose         enable verbose output (used with --dry to describe the commands that would be run)
  -l LOG, --log LOG     writes a log of the install to the specified file
  -r [L4T_REPO], --l4t-repo [L4T_REPO]
                        specify the L4T apt repo (i.e. when using an apt mirror; default is repo.download.nvidia.com/jetson) 
```
4. The dGPU driver install may be verified using the *query* command:
```
$ nvgpuswitch.py query
dGPU (cuda-drivers, 510.73.08-1)
Quadro RTX 6000, 24576 MiB
```

5. After the dGPU drivers have been installed, rebooting the system will complete
   the switch to the dGPU. At this point the Ubuntu desktop will be output via DisplayPort on the dGPU,
   so the display cable must be switched from the onboard HDMI (port 10) to DisplayPort (port 4)
   on the dGPU. 

    **Note** : If the output connection isn't switched before the Clara AGX Developer Kit
    finishes rebooting, the terminal screen will hang during booting.

6. Modify the PATH and LD\_LIBRARY\_PATH . CUDA installs its runtime binaries such as nvcc
into its own versioned path, which is not included by the default $PATH environment variable.
Because of this, attempts to run commands like nvcc will fail on dGPU unless the CUDA 11.6 path
is added to the $PATH variable. To add the CUDA 11.6 path for the current user,
add the following lines to $HOME/.bashrc after the switch to dGPU:
```
export PATH=/usr/local/cuda-11.6/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
```

At this time, the Clara Holoscan SDK is tested and supported only in dGPU mode. Switching back
to iGPU mode after switching to dGPU mode is not recommended.

**Note:** The GPU settings will persist through reboots until it is changed again with *nvgpuswitch.py*.

## Enabling the HDMI Input
The Clara AGX Developer Kit includes an HDMI input (port 11), which is connected internally to
the Jetson CSI interface. Holopack does not configure this CSI interface by default to enable
the HDMI input board, so this configuration must be done manually one time after Holopack
is flashed onto the device. To do this, run the *jetson-io.py* script and select the following
sequence of options to program the CSI connector pins to be compatible with the HDMI input board.

**Note:** if the options are not visible, resize the terminal window to make it taller.

1. Run the script.
```
$ sudo /opt/nvidia/jetson-io/jetson-io.py
```

2. Select “Configure Jetson AGX Xavier CSI Connector.”

3. Select "Configure for compatible hardware". 

4. Select "Jetson Camera HDMI CSI Bridge".

5. Select "Save pin changes".

6. Select "Save and reboot to reconfigure pins".

7. Press any key to reboot.

Once the system has rebooted, operation of the CSI input board can be verified using the v4l2ctl
utility to check that the */dev/video0* device is visible and reports the supported formats:
```
$ sudo apt-get install -y v4l-utils
$ v4l2-ctl -d /dev/video0 --list-formats-ext
ioctl: VIDIOC_ENUM_FMT
	Type: Video Capture

	[0]: 'AR24' (32-bit BGRA 8-8-8-8)
		Size: Discrete 1920x1080
			Interval: Discrete 0.017s (60.000 fps)
		Size: Discrete 1280x720
Interval: Discrete 0.017s (60.000 fps)
```

## Setting up SSD Storage
   **Note:** If the Clara AGX Developer Kit is reflashed with a new Holopack image, 
   the partition table of the m2 drive will not be modified, and the contents of the partition
   will be retained. In this case, the Create Partition steps can be skipped; however, the
   Mount Partition steps should be followed again in order to remount the partition.

```
$ ls -l /dev/disk/by-path/platform-14100000.pcie-pci-0001\:01\:00.0-ata-1
lrwxrwxrwx 1 root root 9 Jan 28 12:24 /dev/disk/by-path/platform-14100000.pcie-pci-0001:01:00.0-ata-1 -> ../../sda
```

## Create a Partition
1. Launch the fdisk utility:
```
$ sudo fdisk /dev/sda
```

2. Create a new primary partition. Use the command ‘n’, then accept the defaults
   (press enter) for the next four questions to create a single partition that uses the entire drive:

```
Command (m for help): n
 Partition type
    p   primary (0 primary, 0 extended, 4 free)
    e   extended (container for logical partitions)
 Select (default p):

 Using default response p.
 Partition number (1-4, default 1):
 First sector (2048-488397167, default 2048):
 Last sector, +sectors or +size{K,M,G,T,P} (2048-488397167, default 488397167):

 Created a new partition 1 of type 'Linux' and of size 232.9 GiB.
```

3. Write the new partition table and exit using the ‘w’ command:
```
Command (m for help): w
 The partition table has been altered.
 Calling ioctl() to re-read partition table.
 Syncing disks.
```

4. Initialize the ext4 filesystem on the new partition:
```
$ sudo mkfs -t ext4 /dev/sda1
 mke2fs 1.44.1 (24-Mar-2018)
 Creating filesystem with 486400 4k blocks and 121680 inodes
 Filesystem UUID: c3817b9c-eaa9-4423-ad5b-d6bae8aa44ea
 Superblock backups stored on blocks:
   32768, 98304, 163840, 229376, 294912

 Allocating group tables: done
 Writing inode tables: done
 Creating journal (8192 blocks): done
 Writing superblocks and filesystem accounting information: done
```

## Mount the Partition

1. Create a directory for the mount point.These instructions will use the path /media/m2,
   but any path may be used if preferred.
```
$ sudo mkdir /media/m2
```

2. Determine the UUID of the new partition.The UUID will be displayed as a symlink
   to the /dev/sda1 partition within the /dev/disk/by-uuid directory. For example, the following
   output shows that the UUID of the /dev/sda1 partition is 4b2bb292-a4d8-4b7e-a8cc-bb799dfeb925:
```
$ ls -l /dev/disk/by-uuid/ | grep sda1
lrwxrwxrwx 1 root root 10 Jan 28 10:05 4b2bb292-a4d8-4b7e-a8cc-bb799dfeb925 -> ../../sda1
```

3. Using the mount path and the UUID from the previous steps, add the following line to
   the end of /etc/fstab:
```
UUID=4b2bb292-a4d8-4b7e-a8cc-bb799dfeb925 /media/m2 ext4 defaults 0 2
```

4. Mount the partition. The /etc/fstab entry above will mount the partition automatically
   at boot time. To mount the partition immediately without rebooting instead, use the
   mount command (and df to verify the mount):
```
$ sudo mount -a
$ df -h /dev/sda1
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       229G  5.6M  229G   0% /media/m2
```

5. Use the "chmod" command to manage file system access permissions:
```
$ sudo chmod -R 777 /media/m2
```

## Setting Up Docker and Docker Storage on SSD

1. Install Docker if it has not been installed on your system:
```
$ sudo apt-get update
$ sudo apt-get install -y docker.io docker-buildx
```

2. Create a Docker data directory on the new m.2 SSD partition. This is where Docker will
   store all of its data, including build cache and container images. These instructions
   use the path */media/m2/docker-data*, but you can use another directory name if preferred.
```
$ sudo mkdir /media/m2/docker-data
```

3. Configure Docker by writing the following to */etc/docker/daemon.json*:
```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia",
    "data-root": "/media/m2/docker-data"
}
```

4. Restart the Docker daemon:
```
$ sudo systemctl daemon-reload
$ sudo systemctl restart docker
```

5. Add the current user to the Docker groupso Docker commands can run without *sudo*.
```
# Create the docker group.
$ sudo groupadd docker
# Add your user to the docker group.
$ sudo usermod -aG docker $USER
# Activate the changes to groups. Alternatively, reboot or re-login.
$ newgrp docker 
```

6. Verify that you can run a "hello world" container.
```
$ docker run hello-world
```

### Install drivers

Open Software & Updates and select *Using NVIDIA driver metapackage from nvidia-driver-535 (proprietary, tested)*

### Validate Version

Run `nvidia-smi` after reboot to confirm your driver version.

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.06             Driver Version: 535.183.06   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Quadro RTX 6000                On  | 00000000:09:00.0  On |                  Off |
| 33%   34C    P8              16W / 260W |     51MiB / 24576MiB |      1%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
```

### Install pip
```
sudo apt-get install python3-pip
```

### Install VScode

Download the arm64 deb and install as instructed

### Holohub 

* Clone Holohub into /home/ivan
```
git clone https://github.com/nvidia-holoscan/holohub.git
cd holohub
./dev_container build
```

### Install dependencies
```
cd /media/m2/data/ba_hyperspectral_segmentation/src
pip install -r requirements.txt

