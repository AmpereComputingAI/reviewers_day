FROM ubuntu:22.04
ARG GH_USER=jan-grzybek-ampere
ARG GH_TOKEN
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y build-essential cmake vim wget git numactl libopenblas-dev pkg-config python3 python3-pip libnuma-dev clang
RUN mkdir /workspace
RUN mkdir /llm
COPY batched_bench.patch /tmp/batched_bench.patch
RUN cd /workspace && git clone -b b3086 https://github.com/ggerganov/llama.cpp.git && cd llama.cpp && git apply /tmp/batched_bench.patch && make -j && mv /workspace/llama.cpp/batched-bench /llm/
RUN rm -R /workspace
