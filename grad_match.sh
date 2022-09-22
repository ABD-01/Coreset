python src/grad_match.py --topn 5000 --dataset CIFAR100 --model ResNet18 --per_class 1 --with_train 1 
python src/grad_match.py --topn 5000 --dataset CIFAR100 --model ResNet18 --per_class 1 --with_train 0
# python src/grad_match.py --topn 5000 --dataset CIFAR100 --model ResNet18 --per_class 0 --with_train 1
# python src/grad_match.py --topn 5000 --dataset CIFAR100 --model ResNet18 --per_class 0 --with_train 0

python src/grad_match.py --topn 500 --dataset CIFAR10 --model ResNet18 --per_class 1 --with_train 1 
python src/grad_match.py --topn 500 --dataset CIFAR10 --model ResNet18 --per_class 1 --with_train 0
python src/grad_match.py --topn 500 --dataset CIFAR10 --model ResNet18 --per_class 0 --with_train 1
python src/grad_match.py --topn 500 --dataset CIFAR10 --model ResNet18 --per_class 0 --with_train 0

python src/grad_match.py --topn 500 --dataset SVHN --model ResNet18 --per_class 1 --with_train 1 
python src/grad_match.py --topn 500 --dataset SVHN --model ResNet18 --per_class 1 --with_train 0
python src/grad_match.py --topn 500 --dataset SVHN --model ResNet18 --per_class 0 --with_train 1
python src/grad_match.py --topn 500 --dataset SVHN --model ResNet18 --per_class 0 --with_train 0

python src/grad_match.py --topn 500 --dataset MNIST --model ResNet18 --per_class 1 --with_train 1 
python src/grad_match.py --topn 500 --dataset MNIST --model ResNet18 --per_class 1 --with_train 0
python src/grad_match.py --topn 500 --dataset MNIST --model ResNet18 --per_class 0 --with_train 1
python src/grad_match.py --topn 500 --dataset MNIST --model ResNet18 --per_class 0 --with_train 0

git add .
git commit -m "Ran grad_match.py for CIFAR100, CIFAR10, SVHN and MNIST"
git push origin colab