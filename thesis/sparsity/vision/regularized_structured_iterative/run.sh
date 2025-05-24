#!/usr/bin/env bash

mkdir -p results
python -m sparsity.vision.regularized_structured_iterative.cifar10 \
    --model resnet20 \
    --pretrained_path resnet20.pth \
    --num_classes 10 \
    --batch_size 128 \
    --test_batch_size 512 \
    --train_workers 4 \
    --stats_file results/stats.json