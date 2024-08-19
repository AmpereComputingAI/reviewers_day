set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

docker run --privileged=true --name torch -d -m 480g -v $SCRIPT_DIR:/benchmark -it amperecomputingai/pytorch:1.10.0
docker exec -i torch bash -c "export DEBIAN_FRONTEND=noninteractive; bash /benchmark/model_zoo_dev/setup.sh"
