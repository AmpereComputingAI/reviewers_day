set -e

python3 power/log_power.py -s lmsensors_wrapper &
python3 run.py -m Meta-Llama-3-8B-Instruct.Q4_K_M.gguf Meta-Llama-3-8B-Instruct.Q8_0.gguf Meta-Llama-3-8B-Instruct.Q4_K_4.gguf Meta-Llama-3-8B-Instruct.Q8R16.gguf -t 8 12 16 24 32 48 64 96 192 -b 1 2 4 8 16 32 -p 128 -r 0-191 -d amperecomputingai/llama.cpp:1.2.6
rm -f /tmp/log_power
