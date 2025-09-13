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

The running results are automatically saved as .csv files (e.g., nn.csv, sngp.csv) in the /results directory.
The trained model will be saved in the /models directory.

If you want to train SNGP, simply add the flag (and adjust the epochs):
```setup
python train.py --sngp --epochs 15
```

If you want to train deep ensembles, simply add the flag (and adjust the  epochs):
```setup
python train.py --ensemble --epochs 15
```

After training, you can run evaluation.py to get all other results.

```setup
python evaluation.py 
```
