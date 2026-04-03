# HETA Lite - Docker Setup Guide (GPU)

## Prerequisites

1. **Install Docker Desktop**
   - Download from https://www.docker.com/products/docker-desktop/
   - Install and launch Docker Desktop
   - Wait until the Docker icon in the system tray shows "Docker Desktop is running"

2. **NVIDIA GPU drivers** - Install from https://www.nvidia.com/drivers

3. **NVIDIA Container Toolkit** - Follow the install guide at https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

4. **Clone the repository**
   ```bash
   git clone https://github.com/anncy0413/SeniorProject-HETA-Lite.git
   cd SeniorProject-HETA-Lite
   ```

---

## Build and Run

```bash
docker compose -f docker/docker-compose.yml build --progress=plain
docker compose -f docker/docker-compose.yml up
```

> `--progress=plain` shows full build output including download progress bars.
> The first build takes ~15-20 minutes depending on internet speed (PyTorch GPU is ~900 MB).

## Access the App

Once you see output like:
```
Running on local URL:  http://0.0.0.0:7860
```

Open your browser and go to: **http://localhost:7860**

The first launch will also download the Qwen2.5-1.5B model (~3 GB). This is cached for future runs.

---

## Stopping the App

Press `Ctrl+C` in the terminal, or run:

```bash
docker compose -f docker/docker-compose.yml down
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "port 7860 already in use" | Stop any other service on that port, or edit the compose file to change `"7860:7860"` to `"7861:7860"` and open http://localhost:7861 instead |
| Build fails downloading PyTorch | Re-run the build command — Docker caches completed steps so it picks up where it left off |
| "out of memory" error at runtime | The model needs ~4 GB RAM. Close other applications and retry |
| GPU not detected | Verify `nvidia-smi` works in your terminal. If not, reinstall NVIDIA drivers |
| No download progress shown | Make sure you include `--progress=plain` in the build command |
