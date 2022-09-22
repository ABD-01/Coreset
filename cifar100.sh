python src/grad_match.py --topn 5000 --dataset CIFAR100 --model ResNet18 --per_class 1 --with_train 1 
python src/grad_match.py --topn 5000 --dataset CIFAR100 --model ResNet18 --per_class 1 --with_train 0

## N=5000
# without train
# python src/train.py --dataset CIFAR100 --topn 5000 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --class_balanced 1
python src/train.py --dataset CIFAR100 --topn 5000 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --per_class 1 
# python src/train.py --dataset CIFAR100 --topn 5000 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1
# with train
# python src/train.py --dataset CIFAR100 --topn 5000 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 --class_balanced 1
python src/train.py --dataset CIFAR100 --topn 5000 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 --per_class 1 
# python src/train.py --dataset CIFAR100 --topn 5000 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1

## N=1000
# without train
# python src/train.py --dataset CIFAR100 --topn 1000 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --class_balanced 1
python src/train.py --dataset CIFAR100 --topn 1000 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --per_class 1 
# python src/train.py --dataset CIFAR100 --topn 1000 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1
# with train
# python src/train.py --dataset CIFAR100 --topn 1000 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 --class_balanced 1
python src/train.py --dataset CIFAR100 --topn 1000 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 --per_class 1 
# python src/train.py --dataset CIFAR100 --topn 1000 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1

## N=100
# without train
# python src/train.py --dataset CIFAR100 --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --class_balanced 1
python src/train.py --dataset CIFAR100 --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --per_class 1 
# python src/train.py --dataset CIFAR100 --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1
# with train
# python src/train.py --dataset CIFAR100 --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 --class_balanced 1
python src/train.py --dataset CIFAR100 --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 --per_class 1 
# python src/train.py --dataset CIFAR100 --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1