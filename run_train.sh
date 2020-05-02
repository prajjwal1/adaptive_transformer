export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="0-55"
export KMP_AFFINITY=granularity=fine,proclist=[0-55],explicit
export OMP_NUM_THREADS=24



# To run layerdrop
#numactl --cpunodebind=0 --membind=0 python3 train.py --bs=128 --layerdrop --tiny --epochs=1 #--load_model=pretrained/model_LXRT

numactl --cpunodebind=0 --membind=0 python3 train.py --bs=128 --layerdrop --tiny --test --load_model=layerdrop_955_ldrop_1_2_1.pth

#numactl --cpunodebind=0 --membind=0 python train.py --bs=128 --test --adaptive --load_model=adaptive_6910

