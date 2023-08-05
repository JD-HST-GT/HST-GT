# **HST-GT**: **H**eterogeneous **S**patial-**T**emporal **G**raph **T**ransformer for Delivery Time Estimation in Warehouse-Distribution Integration E-Commerce

## The Paper has been accepted by CIKM'23!


## Run on google colab (`recommend`)

Our codes, dataset, model and other data are stored in Google Drive （ https://drive.google.com/drive/folders/18MWYE5LteFZLRx-rCHS53ezZTmvvgTU5?usp=sharing ）, and you can train the HST-GT model with Colab.





Train the HST-GT model:
```python
run train.ipynb
```
with colab, to run the train code successfully, we recommend `colab pro+` and choose the `gpu` option.

If you don't subscribe the Colab Pro+, please use the `continue_train.ipynb`.

Test the HST-GT model
```python
run test.ipynb
```
with colab, to run the test code successfully, we recommend choosing the `gpu` option.



## Run on your gpu server
Create the environment(Cuda 11.3)
```python
conda create --name HSTGT --file requirements.txt
```

Train the HST-GT model:
```python
conda activate HSTGT
python train.py
```

Test the HST-GT model
```python
conda activate HSTGT
run test.ipynb
```
