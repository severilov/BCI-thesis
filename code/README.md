# Usage guide

## Installation

```bash
git clone git@github.com:severilov/master-thesis.git
cd master-thesis
chmod 777 create_venv.sh
./create_venv.sh
source venv/bin/activate
```

## Run train

```bash
python3 train.py -с configs/example.yaml
```
Tensorboard usage
```bash
tensorboard --logdir=logs/["model_name"]/["task_type"] --bind_all
```

## Author
Pavel Severilov \
email : pseverilov@gmail.com
