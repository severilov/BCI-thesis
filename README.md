# Physics-based DL models symmetry

- [Physics-based DL models symmetry](#bayesian-experimental-design-using-regularized-determinantal-point-processes)
  - [Main resources](#main)
  - [Abstract](#abstract)
  - [Results](#results)
  - [Video reviews](#video-reviews-by-pseverilov)
  - [Literature](#literature)
      - [General](#general)
      - [Models](#models)
      - [Symmetry and Noether's theorem](#symmetry-and-noethers-theorem)
      - [Useful code links](#useful-code-links)
      - [Other intresting](#other-intresting)

## Main
* [Master Thesis](https://github.com/severilov/master-thesis/blob/main/doc/Severilov2022MasterThesis_rus.pdf)
* [Master Thesis presentation](https://github.com/severilov/master-thesis/blob/main/pres/Severilov2022MasterThesisPres.pdf)
* [Code for experiments](https://github.com/severilov/master-thesis/tree/main/code)
* [Code Usage](https://github.com/severilov/master-thesis/tree/main/code/README.md)

## Abstract
TODO

## Results
TODO

## Video reviews by P.Severilov
* [Hamiltonian and Lagrangian Neural Networks](https://www.youtube.com/watch?v=Q-b-tKAtPtc&t=76s)
* [Fourier Neural Operator for PDEs](https://www.youtube.com/watch?v=YA3Vb9e5hQI&t=22s)

## Literature
### General
* [book] [Physics-based Deep Learning](https://arxiv.org/abs/2109.05237) ([Book Website](https://physicsbaseddeeplearning.org/intro.html))
* [links] [Differential Equations in Deep Learning](https://github.com/Zymrael/awesome-neural-ode#neural-gdes) (Neural ODEs, Neural GDEs, Neural SDEs, Neural CDEs)
* [paper] [A Theoretical Analysis of Deep Neural Networks and
Parametric PDEs](https://link.springer.com/content/pdf/10.1007/s00365-021-09551-4.pdf)
* Overview of Physics-Based Machine Learning:
    * [paper] [Physics-Guided Deep Learning for Dynamical Systems](https://arxiv.org/abs/2107.01272) (!)
    * [paper] [Integrating Machine Learning with Physics-Based Modeling](https://arxiv.org/pdf/2006.02619.pdf)
    * [paper] [Integrating Scientific Knowledge with Machine Learning
for Engineering and Environmental Systems](https://arxiv.org/pdf/2003.04919.pdf)
    * [links] [Neural Operator approaches links](https://zongyi-li.github.io/neural-operator/)
    * (optional) [paper] [Scientific Machine Learning through Physics-Informed Neural Networks: Where we are and What’s next](https://arxiv.org/pdf/2201.05624.pdf)
### Models
* **HNN**: [[paper](https://arxiv.org/abs/1906.01563v1) | [code](https://github.com/greydanus/hamiltonian-nn)] (Hamiltonian Neural Networks)
* **LNN**: [[paper](https://arxiv.org/abs/2003.04630) | [code](https://github.com/MilesCranmer/lagrangian_nns) | [lecture](https://www.youtube.com/watch?v=27ravidF96g)]
(Lagrangian Neural Networks)
* **DeLaN**: [[paper1](https://arxiv.org/abs/1907.04489) [paper2](https://arxiv.org/abs/1907.04490) [paper3](https://arxiv.org/abs/2110.01894) | [code](https://github.com/milutter/deep_lagrangian_networks) | [other_related_projects](http://www.mlutter.eu/projects/)] (Deep Lagrangian Networks)
* **FNO**: [[paper](https://arxiv.org/abs/2010.08895) | [code](https://github.com/zongyi-li/fourier_neural_operator) | [blog_post](https://zongyi-li.github.io/blog/2020/fourier-pde/)]
(Fourier Neural Operator for Parametric Partial Differential Equations)
* **PINO**: [[paper](https://arxiv.org/abs/2111.03794) | [code](https://github.com/devzhk/PINO)] (Physics-Informed Neural Operator for Learning Partial Differential Equations)
* **DeepONet**: [[paper](https://arxiv.org/abs/1910.03193) | [code](https://github.com/lululxvi/deeponet) | [presentation](https://lululxvi.github.io/files/talks/2020SIAMMDS_MS1.pdf)]
(DeepONet: Learning nonlinear operators for identifying differential equations based on the universal approximation theorem of operators)
* **NeuralODE**: [[paper](https://arxiv.org/abs/1806.07366) | [code](https://github.com/rtqichen/torchdiffeq)]
(Neural Ordinary Differential Equations)
* **DiffCoSim**: [[paper](https://arxiv.org/abs/2102.06794) | [code](https://github.com/Physics-aware-AI/DiffCoSim)] (Extending Lagrangian and Hamiltonian Neural Networks with Differentiable Contact Models)
* **PINNs**: [[paper1](https://arxiv.org/abs/1711.10561) [paper2](https://arxiv.org/abs/1711.10566) [paper3](https://www.sciencedirect.com/science/article/pii/S0021999118307125) | [code](https://github.com/janblechschmidt/PDEsByNNs/blob/main/PINN_Solver.ipynb)] (Physics-informed neural networks)
* (???) **PGNN**: [[paper](https://arxiv.org/pdf/1710.11431.pdf) | code] (Physics-guided Neural Networks (PGNN): An Application in Lake Temperature Modeling)

### Symmetry and Noether's theorem
* [theory] [Теоремы Нётер](https://www.wikiwand.com/ru/%D0%A2%D0%B5%D0%BE%D1%80%D0%B5%D0%BC%D0%B0_%D0%9D%D1%91%D1%82%D0%B5%D1%80)
* [paper] [Noether's Learning Dynamics: Role of Symmetry Breaking in Neural Networks](https://arxiv.org/abs/2105.02716)
* [blog_post] [Noether’s Theorem, Symmetries, and Invariant Neural Networks](https://fabianfuchsml.github.io/noether/)
* [paper](https://arxiv.org/abs/2012.04728) | [lecture](https://www.youtube.com/watch?v=xiumcpMqkXM) | [slides](http://www.physicsmeetsml.org/pdf/seminar_kunin-tanaka.pdf) | [blog_post](https://ai.stanford.edu/blog/neural-mechanics/) Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics
* [paper] [Noether: The More Things Change, the More Stay the Same](https://arxiv.org/pdf/2104.05508.pdf)
* [paper] [Interpretable conservation law estimation by deriving the symmetries of dynamics
from trained deep neural networks](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.103.033303)

### Useful code links
<!--  
 * [Experiments comparison of FC NN, LSTM, LNN on pendulum system](https://github.com/gthampak/physinet.io) 
 -->
* [A self-contained tutorial for LNNs.ipynb](https://colab.research.google.com/drive/1CSy-xfrnTX28p1difoTA8ulYw0zytJkq#scrollTo=mhUbF1-vXY-b)
* (???) [Jupyter notebooks with 3 approaches to solve PDEs by NNs](https://github.com/janblechschmidt/PDEsByNNs) (PINNs, Feynman-Kac solver, Deep BSDE solver) ([PDF](https://onlinelibrary.wiley.com/doi/full/10.1002/gamm.202100006))
* [PhyCRNet](https://github.com/isds-neu/PhyCRNet) (Physics-informed convolutional-recurrent neural networks for PDEs)
* Libraries for physics-informed learning with NNs:
    * [DeepXDE](https://github.com/lululxvi/deepxde) ([paper](https://arxiv.org/pdf/1907.04502.pdf) !!!)
    * [PyDEns](https://github.com/analysiscenter/pydens)

### Other intresting:
* [links] [links to many Physics-Based Deep Learning papers](https://github.com/thunil/Physics-Based-Deep-Learning)
* [videos] [Talks on physics and ML](http://www.physicsmeetsml.org/)
* ! [paper] [Unsupervised Learning of Lagrangian Dynamics from Images for Prediction and Control](https://arxiv.org/pdf/2007.01926.pdf)
* ! [paper] [Noether Networks: Meta-Learning Useful Conserved Quantities](https://arxiv.org/pdf/2112.03321.pdf) | [lecture](https://www.youtube.com/watch?v=Xp3jR-ttMfo)
* ! [paper] [Meta-Auto-Decoder for Solving Parametric Partial Differential
Equations](https://arxiv.org/pdf/2111.08823.pdf)
* [paper] [Identifying Physical Law of Hamiltonian Systems via Meta-Learning](https://arxiv.org/abs/2102.11544)
* [paper] [Discovering Symbolic Models from Deep Learning with Inductive Biases](https://arxiv.org/abs/2006.11287)
* [paper] [Visual Interaction Networks: Learning a Physics Simulator from Video](https://proceedings.neurips.cc/paper/2017/file/8cbd005a556ccd4211ce43f309bc0eac-Paper.pdf)
* [paper] [Neural Network Augmented Physics Models for Systems with Partially Unknown Dynamics](https://arxiv.org/pdf/1910.12212.pdf)
* [paper] [Model Reduction And Neural Networks For Parametric PDEs](https://arxiv.org/abs/2005.03180)
<!---  
* PLS: [Roman Isachenko PHD Thesis](https://github.com/r-isachenko/PhDThesis)
--->
