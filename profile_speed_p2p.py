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


def test_all_reduce_time(args):
    array_size = 13107200
    assigned_device = "cuda:{}".format(args.local_rank)
    torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 4 + args.local_rank
    tensor_rand = torch.rand(array_size, device=assigned_device, dtype=torch.float32)
    world_size = dist.get_world_size()
    # accum_list = [torch.zeros_like(tensor_rand) for i in range(world_size)]
    time_list = list()
    start_time_backward = torch.cuda.Event(enable_timing=True)
    stop_time_backward = torch.cuda.Event(enable_timing=True)
    for i in range(100):
        print(i)
        tensor_rand = torch.rand(
            array_size, device=assigned_device, dtype=torch.float32
        )
        start_time_backward.record()
        dist.all_reduce(tensor_rand)
        stop_time_backward.record()
        torch.cuda.synchronize()
        time_taken = start_time_backward.elapsed_time(stop_time_backward)
        time_list.append(time_taken)
        print(time_taken)
    data_dict = dict()
    data_dict["timing_log"] = time_list
    file_name = "k80_single_node_{}_{}.json".format(world_size, args.local_rank)
    with open(file_name, "w") as fout:
        json.dump(data_dict, fout)


def test_p2p_time(args, src_device, dest_device):
    array_size = 13107200
    # assigned_device = "cuda:{}".format(args.local_rank)
    # torch.cuda.set_device(args.local_rank)
    global_rank = args.node_rank * 4 + args.local_rank

    start_time_backward = torch.cuda.Event(enable_timing=True)
    stop_time_backward = torch.cuda.Event(enable_timing=True)
    for i in range(100):
        print(i)
        send_tensor = torch.rand(array_size, device=src_device, dtype=torch.float32)
        recv_tensor = torch.rand(array_size, device=dest_device, dtype=torch.float32)
        start_time_backward.record()
        if args.local_rank == 0:
            dist.send(tensor_rand, dest_device)
        if args.local_rank == 1:
            dist.recv(tensor_rand, src=src_device)
        stop_time_backward.record()
        torch.cuda.synchronize()
        time_taken = start_time_backward.elapsed_time(stop_time_backward)
        time_list.append(time_taken)
        data_dict = dict()
        data_dict["timing_log"] = time_list

        file_name = "k80_single_node_send_{}_recv_{}.json".format(
            src_device, dest_device
        )
        with open(file_name, "w") as fout:
            json.dump(data_dict, fout)


if __name__ == "__main__":
    args = parse_args(argparse.ArgumentParser(description="Large Scale Verification"))
    dist.init_process_group(backend="NCCL", init_method="env://")
    print("Dist init")
    args.model_name = "timing_test_all_reduce_single_machine"
    for i in range(0, 8):
        for j in range(0, 8):
            if i == j:
                continue
            else:
                test_p2p_time(args, i, j)
