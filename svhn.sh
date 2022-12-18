python src/grad_match.py --output_dir $1 --topn 500 --dataset SVHN --model ResNet18 --per_class 1 --with_train 1 --optimizer adam --lr 0.001 
python src/grad_match.py --output_dir $1 --topn 500 --dataset SVHN --model ResNet18 --per_class 1 --with_train 0 --optimizer adam --lr 0.001
python src/grad_match.py --output_dir $1 --topn 500 --dataset SVHN --model ResNet18 --per_class 0 --with_train 1 --optimizer adam --lr 0.001
python src/grad_match.py --output_dir $1 --topn 500 --dataset SVHN --model ResNet18 --per_class 0 --with_train 0 --optimizer adam --lr 0.001

## N=500
# without train
python src/train.py --output_dir $1 --topn 500 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --class_balanced 1
python src/train.py --output_dir $1 --topn 500 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --per_class 1 
python src/train.py --output_dir $1 --topn 500 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18
# with train
python src/train.py --output_dir $1 --topn 500 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --with_train 1 --class_balanced 1
python src/train.py --output_dir $1 --topn 500 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --with_train 1 --per_class 1 
python src/train.py --output_dir $1 --topn 500 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --with_train 1

## N=100
# without train
python src/train.py --output_dir $1 --topn 100 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --class_balanced 1
python src/train.py --output_dir $1 --topn 100 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --per_class 1 
python src/train.py --output_dir $1 --topn 100 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18
# with train
python src/train.py --output_dir $1 --topn 100 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --with_train 1 --class_balanced 1
python src/train.py --output_dir $1 --topn 100 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --with_train 1 --per_class 1 
python src/train.py --output_dir $1 --topn 100 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --with_train 1

## N=10
# without train
python src/train.py --output_dir $1 --topn 10 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --class_balanced 1
python src/train.py --output_dir $1 --topn 10 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --per_class 1 
python src/train.py --output_dir $1 --topn 10 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18
# with train
python src/train.py --output_dir $1 --topn 10 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --with_train 1 --class_balanced 1
python src/train.py --output_dir $1 --topn 10 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --with_train 1 --per_class 1 
python src/train.py --output_dir $1 --topn 10 --dataset SVHN --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --with_train 1