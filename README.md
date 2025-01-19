# A unified uncertainty-informed approach for risk management of deep learning models in the open world

The official implementation of the paper ["A unified uncertainty-informed approach for risk management of deep learning models in the open world"].

&nbsp;
![The RFF-networks](RFF-networks.png)
![The risk analysis flowchart](Risk-analysis-flowchart.png)


To install requirements:
```setup
pip install -r requirements.txt
```

Run example:

If you want to train SNGP, simply add the flag (and adjust the epochs):
python train.py --sngp --epochs 20

If you want to train deep ensemble, simply add the flag (and adjust the  epochs):
python train.py --ensemble --epochs 10
