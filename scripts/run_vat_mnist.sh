#!/usr/bin/env bash

# eps=0.3,0.5,0.8 can be about 98%

python pytorch_vat_mnist_semi.py \
  --gpu-id=7 --num-epochs=100 \
  --trainer=vat --xi=0.000001 --eps=0.3 \
  --top-bn \
  --vis
  