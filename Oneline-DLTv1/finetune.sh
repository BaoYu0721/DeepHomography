JOB_NAME=finetune_deep_homo

srun -p VI_AIC_1080TI --gres=gpu:4 \
   --job-name=${JOB_NAME} \
python -u train.py --gpus 4 --cpus 8 --lr 0.000064 --batch_size 32 --finetune True \
   2>&1 | tee logs/${JOB_NAME}.txt &
