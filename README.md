# Inference steps

1. create virtual environment and install the requirements
```
conda create --name ALEX_3 python=3.9 -y 
conda activate ALEX_3
pip install -r requirements.txt

```
2. download [dataset](https://drive.google.com/file/d/1zkyu6uXCLCKZo48Ei3XCUXv95_mDd136/view?usp=sharing)
unzip it in the ./data folder
3. run command in virtual environment
cd to this folder and run the command below in your annaconda

```
python evaluation.py --data data/INCART_10s/INCART_10s.data --weights weights\QRS_AAMI_10s_Re-59-epoch-0.944722ap-model.pth
```

4. save the evaluation result
copy the final output of the terminal like the following:
<img src="./result_sample.png" alt="result_sample">
