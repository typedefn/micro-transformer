# A custom Transformer architecture utilizing ROCm API implemented for AMD GPUs as an experiment

Supports training and inference, custom hyper parameters are set on the command line prior to run.

NOTE: Still under heavy development not 100% on GPU, utilizing CPU as well.

In order to compile you must execute with the hipcc compiler.
Also supports openmp for multi-threading, pass in -fopenmp if you want to use it.

```bash
hipcc Transformer.cpp -o transformer -std=c++17 -lrocblas -fopenmp
```

# This will start training with default hyper-parameters
```bash
./transformer --train   
```

# Inference with the default trained hyper-parameters
```bash
./transformer --inference
```


