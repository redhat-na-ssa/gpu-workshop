# Accelerate ML Model Training on GPUs with Parallel Processing 

(A work in progress)

## Overview

Why is parallel processing useful for AI/ML?

- When training execution run times on a single GPU are prohibitive.
- When data sizes exceed the memory of a single GPU.
- When model (neural network) sizes exceed the memory of a single GPU.

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
  
### Data Parallel Deployment Example

- 2 AWS Instances each containing:
  - 16 vCPUs, 512GB RAM
  - 4 NVIDIA Tesla M60s (8GB)
  - RHEL 9.2/NVIDIA/Pytorch

- Training Dataset: Fashion MNIST Image Classifier
- [Deep Neural Network implemented using the Pytorch Distributed Framework](https://pytorch.org/tutorials/beginner/dist_overview.html)

Procedure for running a simple example on (2) nodes with multiple GPUs on each node.

Python setup for all nodes
```
pip install pip torch numpy torchvision torch_optimizer -U
```
Running the example:

Edit `04-pt-data-parallel.py`, goto line 51 and change os.environ['MASTER_ADDR'] = 'localhost'
to the hostname of your main node.

Run the following example on each node by changing the only the `--node-id=n` argument. Pass in `--node-id=n`
for the main and increment for each worker node.

```
$ python 04-pt-data-parallel.py --num-nodes=2 --node-id=0 --num-gpus=4 --target-accuracy=0.75 --batch-size=16
```
#### Data Parallel Results

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

- Scaling begins to diminish approaching 4 GPUs. This *could* be architecture dependent (intra GPU to CPU bandwidth, etc) or a function of training hyper-parameters (requires additional testing). 
- The gap between the curves represents the communication overhead or the cost associated with updating the weights during gradient descent between the main and worker nodes.

### Openshift/Kubernetes Microservices (future work)
  - How
    - Deploy python scripts as Kubernetes jobs via Helm charts
  - Value:
    - A central platform for large scale ML model training.
    - Maximize GPU efficiency
      - GPUs are locked only during job execution.
  - Add Grafana to monitor metrics.
  - Use machine set auto-scalers
  - Investigate check pointing.

This work is based on NVIDIA's excellent [Data Parallelism Workshop](https://www.nvidia.com/en-us/training/instructor-led-workshops/train-deep-learning-models-on-multi-gpus/)