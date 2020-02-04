export OMP_SCHEDULE=STATIC
export OMP_PROC_BIND=CLOSE
export GOMP_CPU_AFFINITY="0-55"
export KMP_AFFINITY=granularity=fine,proclist=[0-55],explicit
export OMP_NUM_THREADS=24
numactl --cpunodebind=0 --membind=0 python fit.py --bs=256 --layerdrop --tiny --epochs=10
