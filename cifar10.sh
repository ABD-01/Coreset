# python src/train.py --config configs/cifar10/cifar10.yml --dataset cifar10 --topn $1 --with_train --temp
python src/train.py --config configs/cifar10/cifar10.yml --dataset cifar10 --topn $1 --class_balanced --with_train --temp
python src/train.py --config configs/cifar10/cifar10.yml --dataset cifar10 --topn $1 --per_class --with_train --temp

# python src/train.py --config configs/cifar10/cifar10.yml --dataset cifar10 --topn $1 --random
# python src/train.py --config configs/cifar10/cifar10.yml --dataset cifar10 --topn $1 --random --class_balanced
