# PyTorch multi-process throughput benchmarks

**Instructions assume a Debian based Linux platform.**
```bash
sudo apt install -y docker.io lm-sensors
sudo bash setup_docker.sh
nohup sudo bash run.sh
```
Benchmarks will take few hours in default setting, going over various combinations of n_proc x n_threads x batch_size.
After they complete you will find .csv files with results under model_zoo_dev/results/csv_files/ directory. 
