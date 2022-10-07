#!/bin/bash 

FEATURES=('att_standardised_combined_features' 'standardised_combined_features' 'eng_features' 'att_eng_features' 'comp_centred_soundings')
COV=('full' 'tied' 'diag' 'spherical')

for feature in "${FEATURES[@]}"
do
    for cov in "${COV[@]}"
    do
        for n in {4..11}
        do  
            nohup nice -n 10 ./GMTest.py $1 -f $feature -c $cov -n $n &> ../gmtest/out.log 
        done
    done
done