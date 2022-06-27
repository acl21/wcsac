# Worst-Case Soft Actor-Critic (WCSAC) implementation in PyTorch

This is PyTorch implementation of Worst-Case Soft Actor-Critic (WCSAC) [[Page]](https://ojs.aaai.org/index.php/AAAI/article/view/17272) [[PDF]](https://www.st.ewi.tudelft.nl/mtjspaan/pub/Yang21aaai.pdf). This repository is built on top [PyTorch SAC](https://github.com/denisyarats/pytorch_sac) by Denis Yarats and Ilya Kostrikov. You can find the official implementation in TensorFlow [here](https://github.com/AlgTUDelft/WCSAC).

If you use this code in your research project please cite us and the original authors as:
```
@misc{pytorch_wcsac,
  author = {Pfrang, Luca and Chandra, Akshay L and Koribille, Sri Harsha and Zhang, Baohe},
  title = {Worst-Case Soft Actor-Critic (WCSAC) implementation in PyTorch},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/acl21/wcsac}},
}

@article{Yang_Simão_Tindemans_Spaan_2021,
  author={Yang, Qisong and Simão, Thiago D. and Tindemans, Simon H and Spaan, Matthijs T. J.},
  title={WCSAC: Worst-Case Soft Actor Critic for Safety-Constrained Reinforcement Learning},
  year={2021},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  volume={35},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/17272}
}
```

## Requirements
TBA

## Instructions
To train an SAC agent on the `Safexp-PointGoal1-v0` task from [Safety Gym](https://openai.com/blog/safety-gym/) run:
```
python train.py env=point_goal_1
```
This will produce `exp` folder, where all the outputs are going to be stored including train/eval logs, tensorboard blobs, and evaluation episode videos. One can attacha tensorboard to monitor training by running:
```
tensorboard --logdir exp
```

## Results
TBA
