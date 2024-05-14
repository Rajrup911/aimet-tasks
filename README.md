# AIMET-Darknet53
[AIMET](https://quic.github.io/aimet-pages/index.html), AI Model Efficiency Toolkit is a library that provides advanced model quantization and compression techniques for trained neural network models. It provides features that have been proven to improve the run-time performance of deep learning neural network models with lower compute and memory requirements and minimal impact on task accuracy.

# Table of Contents
- [Model Source](#Model-Source)
- [Model Description](#Model-Description)
- [Dataset Used](#Dataset-Used)
- [Results](#Results)

## Model Source
Model picked up from 'https://github.com/PRBonn/lidar-bonnetal/tree/master'

## Model Description
1. Model Name: Rangenet (backbone: darknet53)
2. Input Shape: (1, 5, 64, 2048)

## Dataset Used
1. Kitti Pretrained

## Results
- PTQ Methods applied: [CLE, Adaround]
- Adaround Parameters: num_batches = 32, default_num_iterations = 1000

| Type       | FP32  | PTQ |
| ------------- |:-------------:|:-----:|
| Acc avg   | 0.888 | 0.840 | 
| IoU avg   | 0.503 | 0.456 |

- QAT 

## License



### Command to run full pipeline
```zsh
python src/quant.py --config config/config.json
```

