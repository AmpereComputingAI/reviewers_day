set -e

THREADS_TOTAL="0-191"
THREADS_PER_PROCESS="1 2 4 8 16 24 32 48 64 96 192"
BATCH_SIZES="1 2 4 8 16 32 64 128 256"

TORCH_NETWORKS="bert_large_mlperf_squad resnet_50_v1.5 stable_diffusion whisper_medium.en_hf"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python3 $SCRIPT_DIR/model_zoo_dev/power/log_power.py -s lmsensors_wrapper &

python3 $SCRIPT_DIR/model_zoo_dev/benchmark_set.py -m bert_large_mlperf_squad -s offline -f pytorch -p fp16 -t $THREADS_PER_PROCESS -r $THREADS_TOTAL -b 1 2 4 8 --docker_name torch --mzdev_dir /benchmark/model_zoo_dev --debug --pre_cmd="export AIO_SKIP_MASTER_THREAD=1; export ENABLE_AIO_IMPLICIT_FP16=1; source ampere_model_library/set_env_variables.sh;" --timeout=1800
python3 $SCRIPT_DIR/model_zoo_dev/benchmark_set.py -m resnet_50_v1.5 -s offline -f pytorch -p fp16 -t $THREADS_PER_PROCESS -r $THREADS_TOTAL -b 64 128 256 --docker_name torch --mzdev_dir /benchmark/model_zoo_dev --debug --pre_cmd="export AIO_SKIP_MASTER_THREAD=1; export ENABLE_AIO_IMPLICIT_FP16=1; source ampere_model_library/set_env_variables.sh;" --timeout=1800
python3 $SCRIPT_DIR/model_zoo_dev/benchmark_set.py -m stable_diffusion -s offline -f pytorch -p fp16 -t $THREADS_PER_PROCESS -r $THREADS_TOTAL -b 1 2 4 --docker_name torch --mzdev_dir /benchmark/model_zoo_dev --debug --pre_cmd="export AIO_SKIP_MASTER_THREAD=1; export ENABLE_AIO_IMPLICIT_FP16=1; source ampere_model_library/set_env_variables.sh;" --timeout=1800
python3 $SCRIPT_DIR/model_zoo_dev/benchmark_set.py -m whisper_medium.en_hf -s offline -f pytorch -p fp16 -t $THREADS_PER_PROCESS -r $THREADS_TOTAL -b $BATCH_SIZES --docker_name torch --mzdev_dir /benchmark/model_zoo_dev --debug --pre_cmd="export AIO_SKIP_MASTER_THREAD=1; export ENABLE_AIO_IMPLICIT_FP16=1; source ampere_model_library/set_env_variables.sh;" --timeout=1800
python3 $SCRIPT_DIR/model_zoo_dev/benchmark_set.py -m dlrm_torchbench -s offline -f pytorch -p fp16 -t $THREADS_PER_PROCESS -r $THREADS_TOTAL -b 1024 2048 4096 8192 16384 32768 65536 131072 --docker_name torch --mzdev_dir /benchmark/model_zoo_dev --debug --pre_cmd="export AIO_SKIP_MASTER_THREAD=1; export ENABLE_AIO_IMPLICIT_FP16=1; source ampere_model_library/set_env_variables.sh;"

rm /tmp/log_power
