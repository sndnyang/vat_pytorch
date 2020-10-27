# VAT PyTorch implementation

virtual adversarial training

MNIST/SVHN/CIFAR10. All of them almost achieve the same results as in the paper.

**NOTE**: No results  of +entropy minimization 

## Preparation of dataset for semi-supervised learning

On MNIST

```
dataset/download_mnist.sh
```

On CIFAR-10

```
python dataset/cifar10.py --data_dir=./dataset/cifar10/
```

On SVHN

```
python dataset/svhn.py --data_dir=./dataset/svhn/
```

## Semi-supervised Learning without augmentation

On MNIST: you must use GPU -- CPU will use different parameters and it's worse.



On CIFAR-10

```
python pytorhc_train_semisup_dataset.py --data_dir=./dataset/cifar10/ --log_dir=./log/cifar10/ --num_epochs=500 --epoch_decay_start=460 --epsilon=10.0 --method=vat
```

On SVHN

```
python pytorhc_train_semisup_dataset.py --data_dir=./dataset/svhn/ --log_dir=./log/svhn/ --num_epochs=120 --epoch_decay_start=80 --epsilon=2.5 --top_bn --method=vat
```



## Semi-supervised Learning with augmentation

On CIFAR-10

```
python train_semisup.py --data_dir=./dataset/cifar10/ --log_dir=./log/cifar10aug/ --num_epochs=500 --epoch_decay_start=460 --aug_flip=True --aug_trans=True --epsilon=8.0 --method=vat
```

On SVHN

```
python train_semisup.py --data_dir=./dataset/svhn/ --log_dir=./log/svhnaug/ --num_epochs=120 --epoch_decay_start=80 --epsilon=3.5 --aug_trans=True --top_bn --method=vat
```

# theano


[theano code](https://github.com/takerum/vat)

python3.7 Theano==1.0.3+2.g3e47d39ac.dirty , numpy=1.15.4 can work now

Done:

1. support python 2.7 and 3.6 together (visualize_contour.py is based on train_syn.py using the same version of python)

# chainer

[chainer code](https://github.com/takerum/vat_chainer)

update the version of chainer.

[original Chainer code from author](https://github.com/takerum/vat_chainer)

