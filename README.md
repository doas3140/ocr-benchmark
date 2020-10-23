# visa ocr benchmark

## Setup

0. prepare docker-compose.yml file
- if use gpu uncomment `runtime: nvidia`
- make sure volume is set to this dir, ex: `~/ocr-benchmark:/code:rw`

1. run container
```
tmux
docker-compose up
# press CTRL+b, then d to close (to get back use `tmux attach`)
```

2. connect to container
```
docker-compose exec web bash
conda activate ml-server
```

3. download unet model
```
gdown -O models/ --id 1Yyzos8epMesi9tcdbbckKQ9cPE802EU7
```

## Examples

### hyperparameter search
```
cd data
gdown --id 1iHIp2VG_z8JwHe5AFvV2O_Oc9IQeGuud # example data
unzip tresh_example.zip
cd ..
python hyperparameter_search.py
```

### visa ocr
```
cd data
gdown --id 1l0oMaeiFe13jzNCiCOqJjzZxeYsgfJ8s # example data
unzip visa_example.zip
cd ..
python visa_benchmark.py
```
