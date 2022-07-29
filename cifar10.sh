python src/train.py --config src/configs/cifar10.yml --dataset cifar10 --topn $1 --with_train
python src/train.py --config src/configs/cifar10.yml --dataset cifar10 --topn $1 --class_balanced --with_train
python src/train.py --config src/configs/cifar10.yml --dataset cifar10 --topn $1 --per_class --with_train

# python src/train.py --config src/configs/cifar10.yml --dataset cifar10 --topn $1 --random
# python src/train.py --config src/configs/cifar10.yml --dataset cifar10 --topn $1 --random --class_balanced
