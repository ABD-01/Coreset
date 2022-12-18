python src/grad_match.py --output_dir $1 --topn 500 --dataset CIFAR10 --model ResNet18 --per_class 1 --with_train 1 --optimizer adam --lr 0.001 --pretrained 1 
python src/grad_match.py --output_dir $1 --topn 500 --dataset CIFAR10 --model ResNet18 --per_class 1 --with_train 0 --optimizer adam --lr 0.001 --pretrained 1
python src/grad_match.py --output_dir $1 --topn 500 --dataset CIFAR10 --model ResNet18 --per_class 0 --with_train 1 --optimizer adam --lr 0.001 --pretrained 1
python src/grad_match.py --output_dir $1 --topn 500 --dataset CIFAR10 --model ResNet18 --per_class 0 --with_train 0 --optimizer adam --lr 0.001 --pretrained 1

## N=500
# without train
python src/train.py --output_dir $1 --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --class_balanced 1
python src/train.py --output_dir $1 --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --per_class 1 
python src/train.py --output_dir $1 --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1
# with train
python src/train.py --output_dir $1 --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --with_train 1 --class_balanced 1
python src/train.py --output_dir $1 --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --with_train 1 --per_class 1 
python src/train.py --output_dir $1 --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --with_train 1

## N=100
# without train
python src/train.py --output_dir $1 --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --class_balanced 1
python src/train.py --output_dir $1 --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --per_class 1 
python src/train.py --output_dir $1 --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1
# with train
python src/train.py --output_dir $1 --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --with_train 1 --class_balanced 1
python src/train.py --output_dir $1 --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --with_train 1 --per_class 1 
python src/train.py --output_dir $1 --topn 100 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --with_train 1

## N=10
# without train
python src/train.py --output_dir $1 --topn 10 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --class_balanced 1
python src/train.py --output_dir $1 --topn 10 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --per_class 1 
python src/train.py --output_dir $1 --topn 10 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1
# with train
python src/train.py --output_dir $1 --topn 10 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --with_train 1 --class_balanced 1
python src/train.py --output_dir $1 --topn 10 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --with_train 1 --per_class 1 
python src/train.py --output_dir $1 --topn 10 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 1200 --min_lr 0.001 --model ResNet18 --pretrained 1 --with_train 1

# python src/train.py --topn 100 --class_balanced 1 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 -bs 1000
