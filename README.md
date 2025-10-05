# PyTorch Implementation of Robustness via Activation-Sparsity-Oriented Training

## Reproducibility of Experiments

### Synthetic Data

- Perturbation Tests:
  
  Run the following command to train a standard 2-layer ReLU network on synthetic data using the selected sparsity method. The script will then evaluate the trained model under adversarial perturbations (Euclidean norm) and visualize the results.
  * <pre> python synthetic_run.py --mode {l1 | dropout} </pre>
  * specify mode with either "l1" or "dropout" for the desired sparsity regularization method during training.

### Real Data
- Train:

  Run the following command to train a convolutional network on the CIFAR-10 dataset:
  * <pre> python main.py --mode train {optional args: --dropout --layers [conv1,conv2...]} </pre>
  * The default sparsity method is l1 - you can select dropout method by inserting '--dropout' argument.
  * By default, sparsity-oriented-training would be executed for all layers in the model (saving the results for each one). You could specify specific layers on which sparsity would be carried out on using the '--layers' argument.
  * The trained models would be saved at 'trained-models/' directory.

- Adversarial Attacks Evaluation:

  Run the following command to test of the robustness of the trained netowrks against PGD and NES adversarial attacks:
  * <pre> python main.py --mode eval {optional args: --dropout --layers [conv1,conv2...]} </pre>
  * As mentioned above, the default sparsity method is l1, and evaluation is carried out for all trained variants (layers). Insert '--dropout' and/or '--layers [...]' arguments according to your selections.

- Visualization:
  Run the following command to visualize the robustness performance of the trained netowrks against PGD and NES adversarial attacks:
  * <pre> python main.py --mode vis {optional args: --dropout --layers [conv1,conv2...]} </pre>
  * As mentioned above, the default sparsity method is l1, and visualization is carried out for all trained variants (layers). Insert '--dropout' and/or '--layers [...]' arguments according to your selections.

## Acknowledgments
The method is heavily inspired by [Gradient Methods Provably Converge to Non-Robust Networks](https://arxiv.org/abs/2202.04347) for synthetic data settings. Thanks for their wonderful work.
