#!/bin/bash 

FEATURES=('att_standardised_combined_features' 'standardised_combined_features' 'eng_features' 'att_eng_features' 'comp_centred_soundings')

for feature in "${FEATURES[@]}"
do
    for b in {50..55}
    do
        mkdir -p ../bctest/$feature/$b
        for t in $(seq 0.6 0.1 3.2)
        do
            nohup nice -n 10 ./BCTest.py $1 -f $feature -b $b -h $t &> ../bctest/out.log
        done
    done
done