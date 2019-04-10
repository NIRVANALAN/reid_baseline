source r0.2.1
srun --partition VI_AIC_TITANXP --gres=gpu:1 --job-name=Dontkillme python train.py --batchsize 64 --train_all --stride 1 --erasing_p 0.5 --gpu_ids 0 2>&1
python test.py
python evaluate_gpu.py

