# ECE-MAE_team7_final_project



How to run the module:


```bash
uv run --directory ~/quectel ~/quectel/p1_runner/bin/runner.py \
  --device-id rQ2gKIc6 \
  --polaris Li48DbiF \
  --device-port /dev/ttyUSB0 \
| stdbuf -oL python3 -c 'import socket,sys; s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM); [s.sendto(line.encode(), ("127.0.0.1", 10110)) for line in sys.stdin if "LLA=" in line]'
```