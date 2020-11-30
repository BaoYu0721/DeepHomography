JOB_NAME=test_deep_homo

srun -p VI_AIC_1080TI --gres=gpu:4 \
   --job-name=${JOB_NAME} \
python -u test.py \
   2>&1 | tee logs/${JOB_NAME}.txt &
