#!/bin/bash

base_fold=$(pwd)/$(dirname $0)
input_fold=${base_fold}/input
output_fold=${base_fold}/output

mp3="mp3treesim"

function mlted {
    tree1=$1
    tree2=$2
    outfold=$3
    fout=${outfold}/mlted.txt

    conv_tree1=".tree1.mlted.txt"
    conv_tree2=".tree2.mlted.txt"
    python3 ${base_fold}/../convert_to_mlted.py ${tree1} > ${conv_tree1}
    python3 ${base_fold}/../convert_to_mlted.py ${tree2} > ${conv_tree2}

    ${base_fold}/../measures/MLTED/MLTED ${conv_tree1} ${conv_tree2} &> ${fout}

    rm ${conv_tree1} ${conv_tree2}

    tail -2 ${fout} | head -1 | cut -d' ' -f4 > ${fout}.res
}

function stereodist {
    tool=$1     # CASet or DISC
    tree1=$2
    tree2=$3
    mode=$4     # i or u
    outfold=$5

    if [ "${mode}" == "u" ]
    then
	mode="-u"
	fout=${outfold}/${tool}.u.txt
    else
	mode=""
	fout=${outfold}/${tool}.i.txt
    fi

    conv_trees=".trees.nw"
    python3 ${base_fold}/../convert_to_stereodist.py ${conv_trees} ${tree1} ${tree2}
    sed -i "s/()/({})/g" ${conv_trees} # to fake an edge in the edgecase
    python3 ${base_fold}/../measures/stereodist/${tool}.py ${mode} ${conv_trees} -o ${fout} 2> ${fout}.log
    rm ${conv_trees}

    tail -1 ${fout} | cut -f 2 | awk '{printf "%.5f\n", (1 - $1)}' > ${fout}.res
}

function run_all {
    tree1=$1
    tree2=$2
    run_out=$3/$(basename ${tree1} .gv)_$(basename ${tree2} .gv)

    echo -n "# $(basename ${tree1} .gv) vs $(basename ${tree2} .gv): "
    
    mkdir -p ${run_out}

    echo -n " mp3..."
    ${mp3} ${tree1} ${tree2} > "${run_out}/mp3.txt"
    cp ${run_out}/mp3.txt ${run_out}/mp3.txt.res

    echo -n " MLTED..."
    mlted ${tree1} ${tree2} ${run_out}

    echo -n " CASet (i)..."
    stereodist CASet ${tree1} ${tree2} i ${run_out}

    echo -n " CASet (u)..."
    stereodist CASet ${tree1} ${tree2} u ${run_out}

    echo -n " DISC (i)..."
    stereodist DISC ${tree1} ${tree2} i ${run_out}

    echo " DISC (u)..."
    stereodist DISC ${tree1} ${tree2} u ${run_out}
}

function run {
    run_name=$1

    echo "### Running ${run_name} ###"

    trees=($(ls ${input_fold}/${run_name}/*.gv))

    out=${output_fold}/${run_name}

    for i in $(seq 0 $(( ${#trees[@]}-1 )))
    do
	tree1=${trees[$i]}
	for j in $(seq $i $(( ${#trees[@]}-1 )))
	do
	    tree2=${trees[$j]}
	    run_all ${tree1} ${tree2} ${out}
	done
    done
}

function summarize {
    run_name=$1

    echo "### Summarizing ${run_name} ###"
    
    for res in mp3 mp3mu mp3mi mlted CASet.i CASet.u DISC.i DISC.u
    do
	python3 ${base_fold}/build_matrix.py ${output_fold}/${run_name}/"*"/${res}.txt.res > ${output_fold}/${run_name}/${res}.csv
    done

    echo ""
}

# Real trees
runs="gerlinger eirew"
for run_name in ${runs}
do
    run ${run_name}
    summarize ${run_name}
done

# Custom tree (edge case)
edge=${input_fold}/eirew/edge.gv
lichee=${input_fold}/eirew/lichee.gv
mipup=${input_fold}/eirew/mipup.gv
out=${output_fold}/eirew_edge/

echo "### Running eirew_edge ###"
run_all ${edge} ${lichee} ${out}
run_all ${edge} ${mipup} ${out}
