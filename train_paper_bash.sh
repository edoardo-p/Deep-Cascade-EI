#!/usr/bin/env bash

python3 demo_train.py --gpu 0 --task 'ct' --ct-views 50 --epochs 5000 --lr 5e-4 --batch-size 2 --ei-trans 5 --ei-alpha 100 --schedule 2000 3000 4000
