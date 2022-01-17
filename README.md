
# Independent Innovation Analysis (IIA)

This code is the official implementation of

Morioka, H., Hälvä H., Hyvärinen A., Independent Innovation Analysis for Nonlinear Vector Autoregressive Process. Proceedings of The 24th International Conference on Artificial Intelligence and Statistics (AISTATS2021), pp. 1549–1557, 2021.

If you are using pieces of the posted code, please cite the above paper. 


## Requirements

Python3

Pytorch


## Training

To train the model(s) in the paper, run this command:

```train
python iia_training.py
```

Set net_model in the code to either 'igcl' or 'itcl'.

'igcl': train by IIA-GCL framework.

'itcl': train by IIA-TCL framework (needs 'num_segment' parameter).


## Evaluation

To evaluate the trained model, run:

```eval
python iia_evaluation.py
```
