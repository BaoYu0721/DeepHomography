JOB_NAME=train_deep_homo

srun --mpi=pmi2  -p VI_AIC_1080TI -n1 --gres=gpu:2 --ntasks-per-node=1 \
   --job-name=${JOB_NAME} \
python -u train.py --gpus 2 --cpus 8 --lr 0.0001 --batch_size 32 \
   2>&1 | tee logs/${JOB_NAME}.txt &
