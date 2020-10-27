
python pytorch_vat_semi_dataset.py \
  --dataset=svhn  \
  --gpu-id=7 --num-epochs=120 \
  --epoch-decay-start=80   \
  --trainer=vat \
  --data-dir=./data \
  --xi=0.000001 --eps=2.5 \
  --drop=0.5  \
  --vis