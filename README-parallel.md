# Approaches to Accelerate Model Training on GPUs with Parallel Processing 

(Currently a work in progress)

## Overview

Why is parallel processing important?

- Training execution run times on a single GPU are prohibitive
- Data sizes exceed the memory of a single GPU
- Model (neural network) sizes exceed the memory of a single GPU

### Approaches

- Data Parallel
  - One main node
  - Multiple worker nodes
  - The number of available GPUs must be the same on each node
  - The data is partitioned across all GPUs
  - The model network is copied to all GPUs
- Model Parallel (future work)
  - Same topology as above
  - The data is copied to all GPUs
  - The model network is partitioned across all GPUs
  
### Deployment Example

- AWS Instances
  - 16 vCPUs, 512GB RAM
  - 4 NVIDIA Tesla M60s (8GB)
  - RHEL 9.2/NVIDIA/Pytorch

Running a simple example on (2) nodes with multiple GPUs on each node.

Python setup for all nodes
```
pip install pip torch numpy torchvision torch_optimizer -U
```

Example

Data: Fashion MNIST Image Classifier

Framework
- [Pytorch Distributed](https://pytorch.org/tutorials/beginner/dist_overview.html)

Running the example:

Run this on each node by changing the only the `node-id=n` argument.

```
$ python 04-pt-data-parallel.py --num-nodes=1 --node-id=0 --num-gpus=4 --target-accuracy=0.75 --batch-size=16
```
#### Results

Speedup
| GPUs | 1 node | 2 nodes|
| ---- | ------ |--------|
| 2    | 1.75   | 1.4    |
| 4    | 3.4    | 2.75   |
| 8    |        | 4.02   |

Graphs

![Throughput](/images/throughput.png)
![Training](/images/training.png)

Observations

The gap in the curves represents the cost associated with updating the weights during gradient descent between the nodes.

- Openshift/Kubernetes Microservices
  - Deploy as jobs via Helm charts
  - Benefits:
    - GPUs can be shared and only locked when the job runs.

This work is based on NVIDIA's excellent [Data Parallelism Workshop](https://www.nvidia.com/en-us/training/instructor-led-workshops/train-deep-learning-models-on-multi-gpus/)