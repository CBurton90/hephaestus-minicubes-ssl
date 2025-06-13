#!/bin/bash

# Check if a time series/directory argument was provided
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 /path/to/webdataset time-series-length"
    echo ""
    echo "e.g $0 /scratch/SDF25/Hephaestus_Minicubes_v0_webdataset/ 1"
    exit 1
fi

DATASET_DIR="$1"
TS_LENGTH="$2"
modes=("train" "val" "test")

for mode in "${modes[@]}"; do
    if [ "$mode" == "train" ]; then
        submodes=("train_pos" "train_neg")
    else
        submodes=("$mode")
    fi

    for submode in "${submodes[@]}"; do
        base_dir="${DATASET_DIR}${TS_LENGTH}/${submode}"
        pattern="sample-${submode}"
        base_name="${base_dir}/${pattern}"

        index=0

        for file in ${base_name}-*.*; do
            extension="${file##*.}"
            new_name="${base_name}-$(printf "%06d" $index).${extension}"
            mv "$file" "$new_name"
            ((index++))
        done

        echo "Files renamed for pattern ${pattern} successfully."
    done
done