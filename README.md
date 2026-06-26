# Micro-Transformer 

A custom Transformer architecture utilizing ROCm API implemented for AMD GPUs as an experiment.

Supports training and inference, custom hyper parameters are set on the command line prior to execution.

NOTE: Still under heavy development not 100% on GPU, utilizing CPU as well.

Uses character based tokenization.

In order to compile you must execute with the hipcc compiler.
Also supports openmp for multi-threading, pass in -fopenmp if you want to use it.

To utilize 2 threads on the CPU export the following.  Change as needed per hardware specs.

```bash
export OMP_NUM_THREADS=2
```

With HIP_VISIBLE_DEVICES=0 you force the application into a strict single-GPU environment on index 0.

```bash
export HIP_VISIBLE_DEVICES=0
```

```bash
hipcc -O3 Kernels.hip -x hip Transformer.cpp -o transformer -std=c++17 -fopenmp
```

## Training 
This will start training with hyper-parameters and architecture set, will work fine on a 12 VRAM GPU.
```bash
./transformer --train --batch-size 64 --seq-length 256 --embedding-length 256 --num-heads 4 --decoder-layers 4 --weight-decay 0.075 --initial-learning-rate 0.0002 --label-smoothing 0.02 --warmup-epochs 12 --dropout 0.25 --epochs 125 --accumulation-steps 8
```

## Inference
Inference with the trained hyper-parameters and trained weight data.
```bash
./transformer --inference --batch-size 1 --seq-length 256 --embedding-length 256 --num-heads 4 --decoder-layers 4 --load model_best.data
```


