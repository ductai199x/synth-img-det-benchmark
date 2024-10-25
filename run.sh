#!/bin/sh

set -xe

python eval_$1.py run -r $2 -s $3 -rn 1000 -sn 1000 -a $4 -w 0 -o /media/nas2/forensic_self_description/results_competing/$1/$2/$3/$4