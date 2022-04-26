export OUTSCALE=1
export NOISE_VARIANCE=0.0084


for REP in {1..20}; do
    for LENSCALE in {0.3,0.5,1.0,2.0,3.0}; do
        for DIMENSION in {10,50,100}; do
            python generate_ciq_dataset.py --n 110000 --outputscale $OUTSCALE --lengthscale $LENSCALE --noise_variance $NOISE_VARIANCE --dimension $DIMENSION --out .tmp_data/data.npy
            echo "LENSCALE=$LENSCALE, DIMENSION=$DIMENSION, REP=$REP"
            aws s3 cp .tmp_data/data.npy s3://nicholas92457/gaussian-process-array-datasets/ciq_synthetic_var0008/DIM{$DIMENSION}_LENSCALE{$LENSCALE}_{$REP}/data.npy
            rm .tmp_data/data.npy
        done
    done
done