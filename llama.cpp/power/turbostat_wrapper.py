import uuid
import subprocess

# this script always returns power of a single socket (the one that has higher power draw), it also assumes 2P system
tmp_file = f"/tmp/power_{str(uuid.uuid4())}.log"
subprocess.Popen(["turbostat", "--quiet", "--show", "PkgWatt", "--interval", "0.5", "--out", tmp_file])

def sample_power() -> float:
    output = subprocess.check_output(["tail", "-n", "4", tmp_file], text=True).split("\n")
    assert output[0] == "PkgWatt"
    total_power = float(output[1])
    socket_0_power = float(output[2])
    socket_1_power = float(output[3])
    return max(socket_0_power, socket_1_power)
