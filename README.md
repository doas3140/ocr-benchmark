# visa ocr benchmark

## Setup

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
