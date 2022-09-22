python src/grad_match.py --topn 500 --dataset SVHN --model ResNet18 --per_class 1 --with_train 1 
python src/grad_match.py --topn 500 --dataset SVHN --model ResNet18 --per_class 1 --with_train 0
python src/grad_match.py --topn 500 --dataset SVHN --model ResNet18 --per_class 0 --with_train 1
python src/grad_match.py --topn 500 --dataset SVHN --model ResNet18 --per_class 0 --with_train 0

## N=500
# without train
python src/train.py --dataset SVHN --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --class_balanced 1
python src/train.py --dataset SVHN --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --per_class 1 
python src/train.py --dataset SVHN --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1
# with train
python src/train.py --dataset SVHN --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 --class_balanced 1
python src/train.py --dataset SVHN --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 --per_class 1 
python src/train.py --dataset SVHN --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1

## N=100
# without train
python src/train.py --dataset SVHN --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --class_balanced 1
python src/train.py --dataset SVHN --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --per_class 1 
python src/train.py --dataset SVHN --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1
# with train
python src/train.py --dataset SVHN --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 --class_balanced 1
python src/train.py --dataset SVHN --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 --per_class 1 
python src/train.py --dataset SVHN --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1

## N=10
# without train
python src/train.py --dataset SVHN --topn 10 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --class_balanced 1
python src/train.py --dataset SVHN --topn 10 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --per_class 1 
python src/train.py --dataset SVHN --topn 10 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1
# with train
python src/train.py --dataset SVHN --topn 10 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 --class_balanced 1
python src/train.py --dataset SVHN --topn 10 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 --per_class 1 
python src/train.py --dataset SVHN --topn 10 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1