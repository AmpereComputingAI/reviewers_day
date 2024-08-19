# Wrapper for multi-process / batched benchmark of llama.cpp

**Instructions assume a Debian based Linux platform.**
```bash
sudo bash setup.sh
bash download_models.sh
nohup sudo bash run.sh
```
Benchmarks will take few hours in default setting, going over various combinations of n_proc x n_threads x batch_size x prompt_size x model_size.
After they complete you will find .csv files with results in the main directory of this repo. 
