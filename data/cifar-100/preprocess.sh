#!/usr/bin/env bash

echo "Preprocessing raw data"
python preprocess/preprocess.py

NAME="cifar-100" # name of the dataset, equivalent to the directory name


cd ../utils

bash ./preprocess.sh --name $NAME $@

cd ../$NAME
