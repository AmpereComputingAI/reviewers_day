FROM ubuntu:22.04
ARG GH_USER=jan-grzybek-ampere
ARG GH_TOKEN
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y build-essential cmake vim wget git numactl libopenblas-dev libomp-dev pkg-config python3 python3-pip libnuma-dev clang
RUN mkdir /workspace
RUN mkdir /llm
RUN cd /workspace && git clone -b v1.2.6 https://$GH_USER:$GH_TOKEN@github.com/AmpereComputingAI/llama.aio.git && cd llama.aio && CXX=clang++ CC=clang cmake . && CXX=clang++ CC=clang cmake --build . --config Release && mv /workspace/llama.aio/bin/llama-batched-bench /llm/
RUN rm -R /workspace
