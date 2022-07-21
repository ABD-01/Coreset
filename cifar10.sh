python src/train.py --config src/configs/cifar10.yml --dataset cifar10 --topn $1
python src/train.py --config src/configs/cifar10.yml --dataset cifar10 --topn $1 --class_balanced
python src/train.py --config src/configs/cifar10.yml --dataset cifar10 --topn $1 --per_class

python src/train.py --config src/configs/cifar10.yml --dataset cifar10 --topn $1 --random
python src/train.py --config src/configs/cifar10.yml --dataset cifar10 --topn $1 --random --class_balanced
