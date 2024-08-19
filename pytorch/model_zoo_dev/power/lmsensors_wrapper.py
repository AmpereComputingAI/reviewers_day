import os
import subprocess

# keep in mind that this script measures power of 2 sockets if 2 sockets are present
try:
    os.remove("/tmp/sensors.sh")
except FileNotFoundError:
    pass

with open("/tmp/sensors.sh", "w") as f:
    f.write("sensors | grep \"power\" | awk '{print $3}' | awk '{s+=$1} END {print s}'")

def sample_power() -> float:
    return float(subprocess.check_output(["bash", "/tmp/sensors.sh"], text=True))
