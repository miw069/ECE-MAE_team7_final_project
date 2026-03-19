# ECE-MAE_team7_final_project (WORK IN PROGRESS)



How to run the module:



**In a separate terminal run the command below**
*See below how to set up the gps module for this command to work.*
```bash
uv run --directory ~/quectel ~/quectel/p1_runner/bin/runner.py \
  --device-id rQ2gKIc6 \
  --polaris Li48DbiF \
  --device-port /dev/ttyUSB0 \
| stdbuf -oL python3 -c 'import socket,sys; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); [s.sendto(line.encode(), ("127.0.0.1", 10110)) for line in sys.stdin if "LLA=" in line]'
```









## Configuring GPS module

1. Install uv 

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Clone gps repo:
```bash
cd ~
git clone ...
cd ~/quectel
touch pyproject.toml
nano pyproject.toml
```

3. Add the following to the pyproject.toml file:
```toml
[project]
name = "quectel"
version = "0.1.0"
description = "Quectel GPS module"
dependencies = [
    "pyserial",
    "socket",
    "sys",
]
```