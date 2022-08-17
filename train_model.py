import os
import sys
import time
import numpy as np
import argparse
import logging
import json
import math
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torchvision import datasets, transforms
from collections import defaultdict
import torchvision.models as models
from torch.autograd import Variable


def parse_args(parser):
    # parser.add_argument("--arch", default="resnet50", type=str,
    # help="network type")
    # parser.add_argument("--master-ip", type=str, help="Ip address of master")
    parser.add_argument("--local_rank", type=int, help="Rank of the experiment")
    parser.add_argument("--batch-size", type=int, help="Batch size to use")
    parser.add_argument("--dataset-location", type=str, help="Data path")
    parser.add_argument("--loader-threads", type=int, default=2, help="Loader threads")
    # parser.add_argument("--device", type=str, default="cuda:0",
    # help="GPU to use")
    parser.add_argument("--log-file", type=str, default="Log file")
    parser.add_argument("--num-workers", type=int, help="Number of total  workers")
    parser.add_argument(
        "--s3-prefix", type=str, default=None, help="s3-prefix to write"
    )
    parser.add_argument("--node_rank", type=int)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()
    return args


def main_trainer(args, bsize):
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    model = models.__dict__[args.model_name]()
    model.to(assigned_device)
    criterion = torch.nn.CrossEntropyLoss().to(assigned_device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank
    )
    model.train()
    start_time = torch.cuda.Event(enable_timing=True)
    stop_time = torch.cuda.Event(enable_timing=True)
    time_list = list()
    data = torch.randn((bsize, 3, 224, 224))
    target = torch.randint(0, 900, [bsize])
    for batch_idx in range(100):
        print(batch_idx)
        data, target = data.to(assigned_device), target.to(assigned_device)
        output = model(data)
        if args.model_name == "googlenet" and args.model_name == "inception_v3":
            output = output[0]
        loss = criterion(output, target)
        torch.cuda.synchronize()  # let's sync before starting
        start_time.record()
        loss.backward()  # we have the gradients
        stop_time.record()
        torch.cuda.synchronize()
        time_list.append(start_time.elapsed_time(stop_time))
        if batch_idx == 30:
            # file_uploader = s3_utils.uploadFile("large-scale-compression")
            data_dict = dict()
            data_dict["args"] = args.__str__()
            data_dict["timing_log"] = time_list
            file_name = "{}_out_file_{}_bsize_{}_no_nvlink.json".format(
                args.model_name, args.local_rank, bsize
            )
            with open(file_name, "w") as fout:
                json.dump(data_dict, fout)
            # file_uploader.push_file(
            # file_name, "{}/{}".format(args.s3_prefix, file_name)
            # )
            # print("Res 50 done")
            break


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Large Scale Verification"))
    dist.init_process_group(backend="NCCL", init_method="env://")
    print("Dist init")
    # args.model_name = "resnet50"
    # main_trainer(args, 16)
    # main_trainer(args, 32)
    # main_trainer(args, 48)
    # args.model_name = "vgg19"
    # main_trainer(args, 16)
    # main_trainer(args, 32)
    # main_trainer(args, 48)
    # args.model_name = "vgg16"
    # main_trainer(args, 16)
    # main_trainer(args, 32)
    # main_trainer(args, 48)
    # args.model_name = "vgg11"
    # main_trainer(args, 16)
    # main_trainer(args, 32)
    # main_trainer(args, 48)
    # args.model_name = "resnet152"
    # main_trainer(args, 16)
    # main_trainer(args, 32)
    # main_trainer(args, 48)
    # args.model_name = "resnet101"
    # main_trainer(args, 16)
    # main_trainer(args, 32)
    # main_trainer(args, 48)
    # args.model_name = "alexnet"
    # main_trainer(args, 16)
    # main_trainer(args, 32)
    # main_trainer(args, 48)
    # args.model_name = "shufflenet_v2_x1_0"
    # main_trainer(args, 16)
    # main_trainer(args, 32)
    # main_trainer(args, 48)
    args.model_name = "googlenet"
    main_trainer(args, 16)
    main_trainer(args, 32)
    main_trainer(args, 48)
