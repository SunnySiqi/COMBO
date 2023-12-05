# COMBO
Implementation of the paper "A Unified Framework for Connecting Noise Modeling to Boost Noise Detection", [[Arxiv](https://arxiv.org/abs/2312.00827)]

## Requirements
The code has been written using Python3 (3.10.4), run `pip install -r requirements.txt` to install relevant python packages.

## File Structure
Run main.py for all the experiments. Sepcific traintools and trainer will be selected according to different datasets. All the hyperparameters can be modified in 'configs' folder, or can be overwritten by passing the arguments.

### Estimation method implementations are in 'estimation' folder.
+ DualT: 'dualT.py'
+ TVD: 'total_variation.py'
+ ROBOT: 'robot.py'
+ Cluster: 'grow_cluster.py'
### Detection method implementations are in 'detection' folder.
+ FINE/FINE+k: 'selection' folder
+ UNICON/UNICON+k: 'UNICON.py'
### Noise source identification implementation is at 'detection/learn_noise_sources.py'
### Training methods implementations are in the trainers, in 'trainer' folder.
+ CIFAR-10/CIFAR-100: 'synthesized_trainer.py'
+ Animal-10N: 'animal10n_trainer.py'
+ CHAMMI-CP: 'cp_trainer.py'
+ Clothing1M: 'clothing1m_trainer.py'

## Training
Run bash scripts in 'scripts' folder.

## Reference Code
 - DualT: https://github.com/a5507203/dual-t-reducing-estimation-error-for-transition-matrix-in-label-noise-learning
 - TVD: https://github.com/YivanZhang/lio/tree/master/ex/transition-matrix
 - ROBOT: https://github.com/pipilurj/ROBOT/tree/main
 - UNICON: https://github.com/nazmul-karim170/UNICON-Noisy-Label
 - FINE: https://github.com/Kthyeon/FINE_official/
