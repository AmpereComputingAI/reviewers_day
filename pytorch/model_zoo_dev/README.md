# model_zoo_dev
Wrapper of ampere_model_library repo for easy benchmarking & testing

## dockers
For quick AIO evaluation you could use publicly available release dockers:
```
amperecomputingai/tensorflow:latest
amperecomputingai/pytorch:latest
amperecomputingai/onnxruntime:latest
```

## first things first

```bash
git clone --recursive git@github.com:AmpereComputingAI/model_zoo_dev.git
cd model_zoo_dev
bash setup.sh
source ampere_model_library/set_env_variables.sh
```

## learning names of available models at given precision

```bash
python3 test_tf.py -m doesntmatter -p fp32
```

## running TF benchmarks

```bash
AIO_NUM_THREADS=16 python3 benchmark_tf.py -m mobilenet_v2 -p fp32 -b 16 --timeout=15.0
```

## running TF end2end network test

```bash
AIO_NUM_THREADS=16 python3 test_tf.py -m mobilenet_v2 -p fp32 --num_runs=1000
```

## running TFLite benchmarks

```bash
AIO_NUM_THREADS=16 python3 benchmark_tflite.py -m ssd_mobilenet_v2 -p int8 -b 12 --timeout=15.0
```

## running TFLite end2end network test

```bash
AIO_NUM_THREADS=16 python3 test_tflite.py -m ssd_mobilenet_v2 -p int8 -b 32 --num_runs=30
```
# Streamlined Benchmarking
The purpose of this section is to quickly replicate the best possible numbers using AIO on Ampere CPUs. If you are interested in running a more comprehensive benchmark on various machines please refer to this document: https://amperecomputing.atlassian.net/wiki/spaces/AT/pages/2448163480.

## Framework setup
Get the docker container with AIO and in framework of your choice (TF, PyTorch, ONNX) here: https://hub.docker.com/u/amperecomputingai 

start your docker with this command
```
docker run --privileged=true --name ampere_aio --network host -v ~/:/ampere -it amperecomputingai/pytorch:latest
```

## Access to MZD
MZD is available as a tarball which can be downloaded from here: https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/model_zoo_dev.tar.gz

you can use this command:

```
wget https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/model_zoo_dev.tar.gz
```

Next, command below will unpack model zoo dev  into a mzd directory in a directory in which it is run:

```
tar -xzvf model_zoo_dev.tar.gz
```

## MZD setup

to setup MZD please run (remember to be inside a docker container):

```
cd mzd
bash setup.sh
source ampere_model_library/set_env_variables.sh (or source a file provided in terminal after running setup.sh)
```

## List All the Models
If you want to see the list of available models run the following command in MZD:

```
python benchmark_pytorch.py -m models -p fp32
```

## Benchmark
### Single Stream
when running single stream benchmarks you will want to benchmark just batch size of 1. You can supply it like so: -b 1

under -f, --framework specify the framework: tf, pytorch, onnx

under -m, --model_names specify the model.

under -b,--batch_sizes specify the batch size that you want to benchmark, in this case 1

under -p, --precision specify the precision to run: fp32 , fp16 , int8

under -t, --num_threads specify the number of threads, for example -t 80

include the flag --debug to direct all output to the console.

To achieve the best latency result for ResNet50 in PyTorch on Altra Max run this command:

```
python benchmark_ss.py -m resnet_50_v1.5 -f pytorch -p fp32 -b 1 -t 128 --debug
```
Latency mean:   4.84 ms,

### Offline
In offline benchmarks you will want to benchmark a selection of different batch sizes and a combination of number of processes times number of threads

under -f, --framework specify the framework: tf, pytorch, onnx

under -m, --model_names specify the model.

under -b,--batch_sizes specify the batch size that you want to benchmark, in this case 1

under -p, --precision specify the precision to run: fp32 , fp16 , int8

under -r,--threads_range specify the range of threads, for Altra it should be 0-127, you can find it out by running lscpu | grep NUMA

under -n,--num_processes specify the number of processes. 

under -t, --num_threads specify the number of threads, for example -t 80

include the flag --debug to direct all output to the console.

To achieve the best throughput result for ResNet50 in PyTorch on Altra Max run this command:

```
python benchmark_offline.py -m resnet_50_v1.5 -p fp32 -f pytorch -b 32 -t 32 -n 4 -r 0-127 --debug
```
Total throughput: 534.86 fps

## Best Combinations
You need to know the best combination of batch size, numer of threads and number of processes that yields the highest results. We run an extensive benchmark which covers many combinations to achieve this information. If you are interested in any particular model please contact our AI team at Ampere: marcel@amperecomputing.com, jan@amperecomputing.com or dkupnicki@amperecomputing.com 