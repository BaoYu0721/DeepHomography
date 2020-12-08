JOB_NAME=run_one_img

srun -p VI_AIC_1080TI --gres=gpu:1 \
   --job-name=${JOB_NAME} \
python -u runOneImg.py \
   2>&1 | tee logs/${JOB_NAME}.txt &
