# Red Hat GPU Workshop 

## Overview
A workshop focused on using GPUs on Red Hat platforms. 

### Table of Contents

1. [Background](#background)
1. [Installation](#installation) 
- [RHEL 8.5](#RHEL)
    - [Driver Installation](#driver-installation)
    - [Testing](#rhel-testing)
- [Openshift 4.10](#Openshift)
    - [Operator Installation](#operator-installation)
    - [Testing](#openshift-testing)

### Background

Graphics Processing Units (GPUs) were originally invented to allow application developers to program 3D graphics accelerators to render photo realistic images in real time. The key is GPUs accelerate matrix and vector math operations (dot product, cross product and matrix multiplies). It turns out that these math operations are used in many applications besides 3D graphics including high performance computing and machine learning. As a result, software libraries were developed to allow non-graphics or general purpose computing applications to take advantage of GPU hardware.

![non-Shaded Skull](./images/skull.jpg) ![Shaded Skull](./images/skullshaded.jpg)

### Installation

#### RHEL

##### Driver Installation

This workshop is based on using the precompiled
nvidia drivers that match a specific Red Hat kernel release.

If running on AWS EC2, the AWS RHUI repos shoud be disabled. 

Need to check whether the drivers will install against RHUI.

```
systemctl disable choose_repo
yum-config-manager --disable rhel-8-appstream-rhui-rpms 
yum-config-manager --disable rhel-8-baseos-rhui-rpms
yum-config-manager --disable rhui-client-config-server-8
reboot
```

Check that `enabled=0` in `/etc/yum/pluginconf.d/amazon-id.conf`.

```
subscription-manager register --username=user@gmail.com
```

Attach subs and enable repos.

```
subscription-manager attach --auto
subscription-manager repos --enable=rhel-8-for-x86_64-baseos-rpms
subscription-manager repos --enable=rhel-8-for-x86_64-appstream-rpms
yum repolist
```

Follow the docs to [install the nvidia drivers](https://developer.nvidia.com/blog/streamlining-nvidia-driver-deployment-on-rhel-8-with-modularity-streams/)

This should subscribe to the `cuda-rhel8-x86_64` repo.

```
dnf module install nvidia-driver:latest
```

Running nvidia-smi will cause the kernel modules to load.
```
nvidia-smi
```

The expected output should resemble the following.
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   37C    P0    26W /  70W |      0MiB / 15360MiB |     10%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

Confirm the kernel modules are loaded.
```
lsmod|grep nvidia


nvidia_drm             61440  0
nvidia_modeset       1118208  1 nvidia_drm
nvidia_uvm           1085440  0
nvidia              38502400  2 nvidia_uvm,nvidia_modeset
drm_kms_helper        253952  1 nvidia_drm
drm                   573440  4 drm_kms_helper,nvidia,nvidia_drm
```

Install cuda and cudnn packages.

```
yum install cuda libcudnn8 -y
```

##### RHEL Testing

###### Non-container app test

Now the system should be ready to run a gpu workload.

A simple test using Tensorflow.

Create a python environment and install tensorflow.

```
python3 -m venv venv
source venv/bin/activate
pip install pip tensorflow -U
```

Run the script to test the tensorflow devices.
```
python src/tf-test.py
```

Compare the CPU vs. GPU elapsed time in the output.
```
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
Matrix Multiply Elapsed Time: {'CPU': 3.397987127304077, 'GPU': 1.9073705673217773}
```

##### Nvidia Container Toolkit

Install the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#podman)

Configure the toolkit for rootless operation.

##### Containerized app test

The `nvidia-smi` output should be similar to what was reported above.

```
podman run --rm --security-opt=label=disable --hooks-dir=/usr/share/containers/oci/hooks.d/ nvidia/cuda:11.0-base nvidia-smi
```

### Openshift 

#### Operator Installation

Install the NFD operator.

Install the nvidia operator.

##### Openshift Testing

Simple Pod

##### OpenDataHub

Notebook limits
