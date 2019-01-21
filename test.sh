#!/bin/bash

OUT_DIR=output/
mkdir -p $OUT_DIR

echo "-> On CPU"
echo "--> Small sample"
NAME=small_cpu_bool
python runner.py data/grammar_small.txt data/graph_small.txt --run_time -c -o $OUT_DIR/$NAME.txt &&
python compare_answers.py $OUT_DIR/$NAME.txt data/true_small.txt

echo "--> Big sample"
NAME=big_cpu_bool
python runner.py data/grammar_big.txt data/graph_big.txt --run_time -c -o $OUT_DIR/$NAME.txt &&
python compare_answers.py $OUT_DIR/$NAME.txt data/true_big.txt
echo "-----------------"

echo "-> On GPU"
echo "-->Without shared memory"
    for type in bool uint8 uint32; do
    echo "---> Type $type"
    echo "----> Small sample"
    NAME=small_gpu_$type_no_sm
    python runner.py data/grammar_big.txt data/graph_big.txt --run_time -t $type -o $OUT_DIR/$NAME.txt &&
    python compare_answers.py $OUT_DIR/$NAME.txt data/true_small.txt

    echo "----> Big sample"
    NAME=big_gpu_$type_no_sm
    python runner.py data/grammar_big.txt data/graph_big.txt --run_time -t $type -o output/$NAME.txt &&
    python compare_answers.py $OUT_DIR/$NAME.txt data/true_big.txt
    echo "------------------"
done
echo "-->With shared memory"
for type in uint8 uint32; do
    echo "---> Type $type"
    echo "----> Small sample"
    NAME=small_gpu_$type_sm
    python runner.py data/grammar_big.txt data/graph_big.txt --run_time -t $type -s -o $OUT_DIR/$NAME.txt &&
    python compare_answers.py $OUT_DIR/$NAME.txt data/true_small.txt

    echo "----> Big sample"
    NAME=big_gpu_$type_sm
    python runner.py data/grammar_big.txt data/graph_big.txt --run_time -t $type -s -o $OUT_DIR/$NAME.txt &&
    python compare_answers.py $OUT_DIR/$NAME.txt data/true_big.txt
    echo "------------------"
done

