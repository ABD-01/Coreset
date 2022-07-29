python src/train.py --config src/configs/cifar100.yml --dataset cifar100 --topn $1 --with_train
python src/train.py --config src/configs/cifar100.yml --dataset cifar100 --topn $1 --class_balanced --with_train
python src/train.py --config src/configs/cifar100.yml --dataset cifar100 --topn $1 --per_class --with_train

# python src/train.py --config src/configs/cifar100.yml --dataset cifar100 --topn $1 --random
# python src/train.py --config src/configs/cifar100.yml --dataset cifar100 --topn $1 --random --class_balanced
