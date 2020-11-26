JOB_NAME=test_deep_homo

srun --mpi=pmi2  -p VI_AIC_1080TI -n1 --gres=gpu:1 --ntasks-per-node=1 \
   --job-name=${JOB_NAME} \
python -u test.py \
   2>&1 | tee logs/${JOB_NAME}.txt &
