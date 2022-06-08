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

##### Check entitlements.
```
oc new-project gputest
oc run -it --rm --image=registry.access.redhat.com/ubi8:latest test-entitlement -- /bin/sh -c "dnf search kernel-header --showduplicates"
```
Expected output.
```
If you don't see a command prompt, try pressing enter.
Red Hat Universal Base Image 8 (RPMs) - AppStream                                                                                 15 MB/s | 2.6 MB     00:00    
Red Hat Universal Base Image 8 (RPMs) - CodeReady Builder                                                                        177 kB/s |  17 kB     00:00    
================================================================== Name Matched: kernel-header ==================================================================
kernel-headers-4.18.0-348.20.1.el8_5.x86_64 : Header files for the Linux kernel for use by glibc
Session ended, resume using 'oc attach test-entitlement -c test-entitlement -i -t' command when the pod is running
pod "test-entitlement" deleted
```

#### Operator Installation

- Using the Openshift web console, install the **Red Hat Node Feature Discovery (NFD)** operator. Use the default values and it should create a namespace called `openshift-nfd`. 
- Next create an **NodeFeatureDiscovery (NFD)** instance in that same `openshift-nfd` namespace.

This should launch a daemonset.

```
oc get pods -n openshift-nfd
```
```
NAME                                      READY   STATUS    RESTARTS   AGE
nfd-controller-manager-56cc649f75-mj7bn   2/2     Running   0          5m52s
nfd-master-4mzkt                          1/1     Running   0          100s
nfd-master-wr4qd                          1/1     Running   0          100s
nfd-master-zhzkq                          1/1     Running   0          100s
nfd-worker-8tr4j                          1/1     Running   0          100s
nfd-worker-r47qd                          1/1     Running   0          100s
```

- Using the Openshift web console, install the **nvidia operator (v1.10.1)**. It should create a namespace called `nvidia-gpu-operator`. 
- Next create a **cluster policy (CP)** instance in the same `nvidia-gpu-operator` namespace.

Wait for all the pods to have a running status. This could take several minutes.

```
oc get pods -n nvidia-gpu-operator
```

```
NAME                                                  READY   STATUS      RESTARTS   AGE
gpu-feature-discovery-2l9db                           1/1     Running     0          16h
gpu-feature-discovery-4hg4g                           1/1     Running     0          16h
gpu-operator-76bf46dcf8-mtkjc                         1/1     Running     0          16h
nvidia-container-toolkit-daemonset-4dsbh              1/1     Running     0          16h
nvidia-container-toolkit-daemonset-crz6c              1/1     Running     0          16h
nvidia-cuda-validator-lhtjz                           0/1     Completed   0          16h
nvidia-cuda-validator-xghsp                           0/1     Completed   0          16h
nvidia-dcgm-7c9g2                                     1/1     Running     0          16h
nvidia-dcgm-7q4fr                                     1/1     Running     0          16h
nvidia-dcgm-exporter-dw4fv                            1/1     Running     0          16h
nvidia-dcgm-exporter-xnknh                            1/1     Running     0          16h
nvidia-device-plugin-daemonset-7c9m7                  1/1     Running     0          16h
nvidia-device-plugin-daemonset-tbf7x                  1/1     Running     0          16h
nvidia-device-plugin-validator-blp4n                  0/1     Completed   0          16h
nvidia-device-plugin-validator-qsctr                  0/1     Completed   0          16h
nvidia-driver-daemonset-410.84.202203221702-0-9pfhk   2/2     Running     0          16h
nvidia-driver-daemonset-410.84.202203221702-0-wgcnv   2/2     Running     0          16h
nvidia-node-status-exporter-89nsc                     1/1     Running     0          16h
nvidia-node-status-exporter-vfsjz                     1/1     Running     0          16h
nvidia-operator-validator-2lzvj                       1/1     Running     0          16h
nvidia-operator-validator-5s9k9                       1/1     Running     0          16h
```

The daemonset pods will build a driver for each node with a GPU.

```
oc logs nvidia-driver-daemonset-410.84.202204112301-0-gf4t4  -n nvidia-gpu-operator  nvidia-driver-ctr --follow

Tue May 17 19:41:23 UTC 2022 Waiting for openshift-driver-toolkit-ctr container to build the precompiled driver ...
```

Check the logs from one of the `nvidia-cuda-validator` pods.

```
oc logs -n nvidia-gpu-operator nvidia-cuda-validator-qpqcg


cuda workload validation is successful
```

##### Openshift Testing

Client application testing.

Create a project as a cluster-admin user. GPU enabled pods require cluster privileges.
Next, create an application and expose it's service.

```
oc new-project gputest
oc new-app docker.io/tensorflow/tensorflow:latest-gpu-jupyter
oc expose service/tensorflow
```

```
oc get routes

NAME                                  HOST/PORT                                                   PATH   SERVICES     PORT       TERMINATION   WILDCARD
route.route.openshift.io/tensorflow   tensorflow-gputest.apps.ocp-green.dota-lab.iad.redhat.com          tensorflow   8888-tcp                 None
```

Dump the logs of the tensorflow pod to obtain the jupyter lab **token**.
```
[I 20:11:42.236 NotebookApp] Jupyter Notebook 6.4.11 is running at:
[I 20:11:42.236 NotebookApp] http://tensorflow-544f7d6b5b-m8sjg:8888/?token=7f5cfa6a9780fd77594c1e6a45ae88002169e98d87a38580
```

It may be necessary to set the `nvidia.com/gpu=1` limit to ensure the pod get scheduled on a node with a GPU.

```
oc set resources deployment/tensorflow --requests=nvidia.com/gpu=1 --limits=nvidia.com/gpu=1
```

Connect to the tensorflow pod and run a quick GPU test.

```
oc rsh tensorflow-6594894964-8gtz5 

$ python
Python 3.8.10 (default, Mar 15 2022, 12:22:08) 
[GCC 9.4.0] on linux
>>> import tensorflow as tf
>>> tf.config.list_physical_devices()
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'), PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
>>> exit()
$
```

Login to Jupyter and run the `classification.ipynb` notebook.

##### Create a new tensorflow/jupyter app from the [nvidia examples site](https://nvidia.github.io/gpu-operator/).

Simple Pod

##### OpenDataHub

Create an ODH instance in your namespace. For testing purposes, a minimal kfdef that includes `jupyterhub` can be deployed ODH.


- `$ oc new-project my-odh`
- Navigate to:
  - Operators
    - Installed Operators
      - *Important* In the upper left => Set Project: my-odh
      - Open Data Hub Operator
        - Create Instance
          - Configure via: YAML view
            - Add a `- cuda-11.0.3` element to the overlays array in the `notebook-image` section just above the `- addtional` element.
            ```
            - kustomizeConfig:
              overlays:
                - cuda-11.0.3
                - additional
              repoRef:
                name: manifests
                path: jupyterhub/notebook-images
              name: notebook-images
            ```
          - Near the bottom change the tarball version to `v.1.0.9`
          - Name
            - opendatahub
          - Labels 
            - Use default label (i.e. don't put anything in this field)
          - Create

Several images will get pulled and eventually a number of Openshift builds should run to build the cuda enabled notebook images.
The entire Open Data Hub deployment could take up to an hour depending on available resources. When the builds
complete there should be 8 of them.

#### Custom Notebook Limits

Configmaps are used to set custom notebook limits. Apply the following configmap before the launching jupyterhub server.
```
oc apply -f src/jupyterhub-notebook-sizes.yml
```

#### Demos

From within Jupyter, clone the following repo:

[Tensor Flow Examples](https://github.com/tensorflow/docs.git)

These tensorflow notebook examples should run:

- `docs/site/en/tutorials/keras/classification.ipynb`
- `docs/site/en/tutorials/quickstart/beginner.ipynb`
- `docs/site/en/tutorials/quickstart/advanced.ipynb`

