python src/train.py --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 -bs 1000
python src/train.py --topn 500 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 -bs 1000
python src/train.py --topn 500 --class_balanced 1 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 -bs 1000

python src/train.py --topn 100 --per_class 1 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 -bs 1000
python src/train.py --topn 100 --class_balanced 1 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 -bs 1000
python src/train.py --topn 100 --per_class 1 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 -bs 1000

python src/train.py --topn 10 --class_balanced 1 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 -bs 1000
python src/train.py --topn 10 --per_class 1 --scheduler cosineannealingwarmrestarts --T_max 10 --T_mult 2 --epochs 300 --min_lr 0.001 --model ResNet18 --temp 1 --with_train 1 -bs 1000
