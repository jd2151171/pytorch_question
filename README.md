# Environment

To replicate the environment:

- Clone the repository:
```shell
git clone https://github.com/jd2151171/pytorch_question
```

- Create a new environment using Conda (NOTE: I am using conda 4.12.0)
```shell
conda create -n <ENV-NAME> python==3.7.5
```
- Run the following command:
```shell
python -m pip install -r requirements.txt
```
# Execution
To execute the code:
- Enter the following command, alongside your chosen framework (one of "PyTorch", "Keras", or "TensorFlow") and device (one of "gpu" or "cpu"):
```shell
python isolated_lstm_run_test.py <FRAMEWORK> <DEVICE>
```
