python src/train.py --config src/configs/cifar100.yml --dataset cifar100 --topn $1
python src/train.py --config src/configs/cifar100.yml --dataset cifar100 --topn $1 --class_balanced
python src/train.py --config src/configs/cifar100.yml --dataset cifar100 --topn $1 --per_class

python src/train.py --config src/configs/cifar100.yml --dataset cifar100 --topn $1 --random
python src/train.py --config src/configs/cifar100.yml --dataset cifar100 --topn $1 --random --class_balanced
