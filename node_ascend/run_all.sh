#!/bin/bash

rm mp3/results.txt DISC/results.txt CASet/results.txt MLTED/results.txt

for i in $(seq 10 -1 1)
do
   mp3treesim "10.gv" "${i}.gv" >> "mp3/results.txt"
done

python3 ../measures/stereodist/DISC.py "DISC/trees.nw" -o "DISC/results.int"
cat DISC/results.int | tr '\t' ' ' | cut -f11 -d' ' | tail -10 | while read line ; do awk -vline=$line 'BEGIN{printf "%.5f\n", (1 - line)}' ; done | tac > "DISC/results.txt"

python3 ../measures/stereodist/CASet.py "CASet/trees.nw" -o "CASet/results.int"
cat CASet/results.int | tr '\t' ' ' | cut -f11 -d' ' | tail -10 | while read line ; do awk -vline=$line 'BEGIN{printf "%.5f\n", (1 - line)}' ; done | tac > "CASet/results.txt"


for i in $(seq 10 -1 1)
do
    ../measures/MLTED/MLTED "MLTED/10.in" "MLTED/${i}.in" 2> /dev/null | tail -2 | head -1  | cut -f4 -d' ' >> "MLTED/results.txt"
done

python3 plot_ascend.py --csvs mp3/results.txt DISC/results.txt CASet/results.txt MLTED/results.txt --names MP3 DISC CASet MLTED --out desc.pdf
