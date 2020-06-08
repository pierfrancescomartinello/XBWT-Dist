#!/bin/bash

mkdir -p nset
rm "nset/results.txt"

for i in $(seq 1 8)
do
   mp3treesim "1.gv" "${i}.gv" >> "nset/results.txt"
done

mkdir -p DISC
python3 ../../../convert_to_stereodist.py "DISC/trees.nw" 1.gv 2.gv 3.gv 4.gv 5.gv 6.gv 7.gv 8.gv
python3 ../../../measures/stereodist/DISC.py "DISC/trees.nw" -o "DISC/results.int"
cat DISC/results.int | tr '\t' ' ' | cut -f2 -d' ' | tail -8 | while read line ; do awk -vline=$line 'BEGIN{printf "%.5f\n", (1 - line)}' ; done  > "DISC/results.txt"

mkdir -p CASet
python3 ../../../convert_to_stereodist.py "CASet/trees.nw" 1.gv 2.gv 3.gv 4.gv 5.gv 6.gv 7.gv 8.gv
python3 ../../../measures/stereodist/CASet.py "CASet/trees.nw" -o "CASet/results.int"
cat CASet/results.int | tr '\t' ' ' | cut -f2 -d' ' | tail -8 | while read line ; do awk -vline=$line 'BEGIN{printf "%.5f\n", (1 - line)}' ; done > "CASet/results.txt"

mkdir -p MLTED
rm "MLTED/results.txt"

for i in $(seq 1 8)
do
    python3 ../../../convert_to_mlted.py "${i}.gv" > "MLTED/${i}.in"
done

for i in $(seq 1 8)
do
    ./../../../measures/MLTED/MLTED "MLTED/1.in" "MLTED/${i}.in" | tail -2 | head -1  | cut -f4 -d' ' >> "MLTED/results.txt"
done

python3 plot_descend.py --csvs nset/results.txt DISC/results.txt CASet/results.txt MLTED/results.txt --names MP3 DISC CASet MLTED --out slide.pdf