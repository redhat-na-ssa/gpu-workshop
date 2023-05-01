import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# TODO Step 0: Include DDP import statement for convenience
import torch.distributed as dist

# Parse input arguments
parser = argparse.ArgumentParser(description='Fashion MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=40,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--target-accuracy', type=float, default=.85,
                    help='Target accuracy to stop training')
parser.add_argument('--patience', type=int, default=2,
                    help='Number of epochs that meet target before stopping')

# TODO Step 1: Add the following to the argument parser:
# number of nodes (num_nodes, type = int, default = 1), 
# ID for the current node (node_id, type = int, default = 0)
# number of GPUs in each node (num_gpus, type = int, default = 1)
parser.add_argument('--num-nodes', type=int, default=1,
                    help='Number of available nodes/hosts')
parser.add_argument('--node-id', type=int, default=0,
                    help='Unique ID to identify the current node/host')
parser.add_argument('--num-gpus', type=int, default=1,
                    help='Number of GPUs in each node')

args = parser.parse_args()

# TODO Step 2: Compute world size (WORLD_SIZE) using num_gpus and num_nodes
# and specify the IP address/port number for the node associated with 
# the main process (global rank = 0):
# os.environ['MASTER_ADDR'] = 'localhost' 
# os.environ['MASTER_PORT'] = '9956' 
WORLD_SIZE = args.num_gpus * args.num_nodes
os.environ['MASTER_ADDR'] = 'localhost' 
os.environ['MASTER_PORT'] = '9956' 

# Standard convolution block followed by batch normalization 
class cbrblock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(cbrblock, self).__init__()
        self.cbr = nn.Sequential(nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=(1,1),
                               padding='same', bias=False), 
                               nn.BatchNorm2d(output_channels), 
                               nn.ReLU()
        )
    def forward(self, x):
        out = self.cbr(x)
        return out

# Basic residual block
class conv_block(nn.Module):
    def __init__(self, input_channels, output_channels, scale_input):
        super(conv_block, self).__init__()
        self.scale_input = scale_input
        if self.scale_input:
            self.scale = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=(1,1),
                               padding='same')
        self.layer1 = cbrblock(input_channels, output_channels)
        self.dropout = nn.Dropout(p=0.01)
        self.layer2 = cbrblock(output_channels, output_channels)
        
    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.dropout(out)
        out = self.layer2(out)
        if self.scale_input:
            residual = self.scale(residual)
        out = out + residual
        
        return out
    
# Overall network
class WideResNet(nn.Module):
    def __init__(self, num_classes):
        super(WideResNet, self).__init__()
        nChannels = [1, 16, 160, 320, 640]

        self.input_block = cbrblock(nChannels[0], nChannels[1])
        
        # Module with alternating components employing input scaling
        self.block1 = conv_block(nChannels[1], nChannels[2], 1)
        self.block2 = conv_block(nChannels[2], nChannels[2], 0)
        self.pool1 = nn.MaxPool2d(2)
        self.block3 = conv_block(nChannels[2], nChannels[3], 1)
        self.block4 = conv_block(nChannels[3], nChannels[3], 0)
        self.pool2 = nn.MaxPool2d(2)
        self.block5 = conv_block(nChannels[3], nChannels[4], 1)
        self.block6 = conv_block(nChannels[4], nChannels[4], 0)
        
        # Global average pooling
        self.pool = nn.AvgPool2d(7)

        # Feature flattening followed by linear layer
        self.flat = nn.Flatten()
        self.fc = nn.Linear(nChannels[4], num_classes)

    def forward(self, x):
        out = self.input_block(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.pool1(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.pool2(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.pool(out)
        out = self.flat(out)
        out = self.fc(out)
        
        return out

def train(model, optimizer, train_loader, loss_fn, device):
    model.train()
    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        labels = labels.to(device)
        images = images.to(device)
        
        # Forward pass 
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Setting all parameter gradients to zero to avoid gradient accumulation
        optimizer.zero_grad()
        
        # Backward pass
        loss.backward()
        
        # Updating model parameters
        optimizer.step()

def test(model, test_loader, loss_fn, device):
    total_labels = 0
    correct_labels = 0
    loss_total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            # Transfering images and labels to GPU if available
            labels = labels.to(device)
            images = images.to(device)

            # Forward pass 
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            
            # Extracting predicted label, and computing validation loss and validation accuracy
            predictions = torch.max(outputs, 1)[1]
            total_labels += len(labels)
            correct_labels += (predictions == labels).sum()
            loss_total += loss
    
    v_accuracy = correct_labels / total_labels
    v_loss = loss_total / len(test_loader)
    
    return v_accuracy, v_loss

# TODO Step 3: Move all code (including comments) under __name__ == '__main__' to 
# a new 'worker' function that accepts two inputs with no return value: 
# (1) the local rank (local_rank) of the process
# (2) the parsed input arguments (args)
# The following is the signature for the worker function: worker(local_rank, args)
def worker(local_rank, args):
    # TODO Step 4: Compute the global rank (global_rank) of the spawned process as:
    # =node_id*num_gpus + local_rank.
    # To properly initialize and synchornize each process, 
    # invoke dist.init_process_group with the approrpriate parameters:
    # backend='nccl', world_size=WORLD_SIZE, rank=global_rank
    global_rank = args.node_id * args.num_gpus + local_rank 
    dist.init_process_group( 
    backend='nccl',  
    world_size=WORLD_SIZE, 
    rank=global_rank 
    ) 
    
    # TODO Step 5: initialize a download flag (download) that is true 
    # only if local_rank == 0. This download flag can be used as an 
    # input argument in torchvision.datasets.FashionMNIST.
    # Download the training and validation sets for only local_rank == 0. 
    # Call dist.barrier() to have all processes in a given node wait 
    # till data download is complete. Following this, for all other 
    # processes, torchvision.datasets.FashionMNIST can be called with
    # the download flag as false.
    download = True if local_rank == 0 else False
    if local_rank == 0:
        train_set = torchvision.datasets.FashionMNIST("./data", download=download, transform=
                                                    transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.FashionMNIST("./data", download=download, train=False, transform=
                                                   transforms.Compose([transforms.ToTensor()]))  
    dist.barrier()
    
    if local_rank != 0:
        train_set = torchvision.datasets.FashionMNIST("./data", download=download, transform=
                                                    transforms.Compose([transforms.ToTensor()]))
        test_set = torchvision.datasets.FashionMNIST("./data", download=download, train=False, transform=
                                                   transforms.Compose([transforms.ToTensor()]))  

    # TODO Step 6: generate two samplers (one for the training 
    # dataset (train_sampler) and the other for the testing 
    # dataset (test_sampler) with  torch.utils.data.distributed.DistributedSampler. 
    # Inputs to this function include:
    # (1) the datasets (either train_loader_subset or test_loader_subset)
    # (2) number of replicas (num_replicas), which is the world size (WORLD_SIZE) 
    # (3) the global rank (global_rank). 
    # Pass the appropriate sampler as a parameter (e.g., sampler = train_sampler)
    # to the training and testing DataLoader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_set,
    num_replicas=WORLD_SIZE,
    rank=global_rank
    )
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
    test_set,
    num_replicas=WORLD_SIZE,
    rank=global_rank
    )

    # Training data loader
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=args.batch_size, drop_last=True, sampler=train_sampler)
    # Validation data loader
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=args.batch_size, drop_last=True, sampler=test_sampler)

    # Create the model and move to GPU device if available
    num_classes = 10

    # TODO Step 7: Modify the torch.device call from "cuda:0" to "cuda:<local_rank>" 
    # to pin the process to its assigned GPU. 
    # After the model is moved to the assigned GPU, wrap the model with 
    # nn.parallel.DistributedDataParallel, which requires the local rank (local_rank)
    # to be specificed as the 'device_ids' parameter: device_ids=[local_rank]
    device = torch.device("cuda:" + str(local_rank) if torch.cuda.is_available() else "cpu")
    
    # TODO Optional: before moving the model to the GPU, convert standard 
    # batchnorm layers to SyncBatchNorm layers using 
    # torch.nn.SyncBatchNorm.convert_sync_batchnorm.   
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(WideResNet(num_classes)).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define the SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr)

    val_accuracy = []

    total_time = 0

    for epoch in range(args.epochs):
        
        t0 = time.time()
        
        # TODO Step 6.5: update the random seed of the DistributedSampler to change
        # the shuffle ordering for each epoch. It is necessary to do this for
        # the train_sampler, but irrelevant for the test_sampler. The random seed
        # can be altered with the set_epoch method (which accepts the epoch number
        # as an input) of the DistributedSampler. 
        train_sampler.set_epoch(epoch)
        
        train(model, optimizer, train_loader, loss_fn, device)
        
        # TODO Step 8: At the end of every training epoch, synchronize (using dist.barrier())
        # all processes to compute the slowest epoch time. 
        # To compute the number of images processed per second, convert images_per_sec
        # into a tensor on the GPU, and then call torch.distributed.reduce on images_per_sec 
        # with global rank 0 as the destination process. The reduce operation computes the 
        # sum of images_per_sec across all GPUs and stores the sum in images_per_sec in the 
        # master process (global rank 0).
        # Once this computation is done, enable the metrics print statement for only the master process.
        dist.barrier()
        epoch_time = time.time() - t0
        total_time += epoch_time

        images_per_sec = torch.tensor(len(train_loader)*args.batch_size/epoch_time).to(device)
        torch.distributed.reduce(images_per_sec, 0)
        
        v_accuracy, v_loss = test(model, test_loader, loss_fn, device)
        
        # TODO Step 9: average validation accuracy and loss across all GPUs  
        # using torch.distributed.all_reduce. To perform an average operation, 
        # provide 'dist.ReduceOp.AVG' as the input for the op parameter in 
        # torch.distributed.all_reduce.  
        torch.distributed.all_reduce(v_accuracy, op=dist.ReduceOp.AVG)
        torch.distributed.all_reduce(v_loss, op=dist.ReduceOp.AVG)
        val_accuracy.append(v_accuracy)
        
        if global_rank == 0:
            print("Epoch = {:2d}: Cumulative Time = {:5.3f}, Epoch Time = {:5.3f}, Images/sec = {}, Validation Loss = {:5.3f}, Validation Accuracy = {:5.3f}".format(epoch+1, total_time, epoch_time, images_per_sec, v_loss, val_accuracy[-1]))

        if len(val_accuracy) >= args.patience and all(acc >= args.target_accuracy for acc in val_accuracy[-args.patience:]):
            print('Early stopping after epoch {}'.format(epoch + 1))
            break
            
    # TODO Step 10: Within __name__ == '__main__', launch each process (total number of 
    # processes is equivalent to the number of available GPUs per node) with 
    # torch.multiprocessing.spawn(). Input parameters include the worker function, 
    # the number of GPUs per node (nprocs), and all the parsed arguments.
if __name__ == '__main__':
    torch.multiprocessing.spawn(worker, nprocs=args.num_gpus, args=(args,))