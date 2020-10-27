#!/usr/bin/env bash

# eps=10
# In official repo
# Semi-supervised Learning without augmentation On CIFAR-10
# python train_semisup.py --data_dir=./dataset/cifar10/ --log_dir=./log/cifar10/ --num_epochs=500 --epoch_decay_start=460 --epsilon=10.0 --method=vat

python pytorch_vat_semi_dataset.py \
  --dataset=cifar10  \
  --gpu-id=7 --num-epochs=500 \
  --epoch-decay-start=460   \
  --trainer=vat \
  --data-dir=./dataset \
  --xi=0.000001 --eps=10 \
  --drop=0.5  \
  --vis