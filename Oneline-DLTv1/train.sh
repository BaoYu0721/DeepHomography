JOB_NAME=train_deep_homo

srun -p VI_AIC_1080TI --gres=gpu:4 \
   --job-name=${JOB_NAME} \
python -u train.py --gpus 4 --cpus 8 --lr 0.0001 --batch_size 32 \
   2>&1 | tee logs/${JOB_NAME}.txt &
