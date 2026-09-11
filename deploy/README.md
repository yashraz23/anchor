# Phase 3 — serving anchor on k3s with a GPU

Everything here runs on a Windows laptop with an RTX 5070 Ti (12 GB) through
Docker Desktop's WSL2 backend. That environment shapes several decisions, and
they are recorded where they were made rather than summarised away.

## Cluster

k3s runs via k3d, which is genuine k3s inside Docker rather than a substitute.
The stock k3s node image has no NVIDIA container runtime, so
`Dockerfile.k3s-cuda` builds one that does; k3s then auto-detects it and writes
the `nvidia` containerd runtime itself.

```bash
docker build -f deploy/k3s/Dockerfile.k3s-cuda -t anchor/k3s-cuda:v1.35.5 deploy/k3s
k3d cluster create anchor --image anchor/k3s-cuda:v1.35.5 --gpus all \
  --servers 1 --agents 0 --port "8081:80@loadbalancer" \
  --k3s-arg "--disable=traefik@server:0"
```

k3d writes a kubeconfig pointing at `host.docker.internal`, which resolves to
the LAN address here and times out. Point it at loopback instead:

```bash
PORT=$(docker port k3d-anchor-serverlb 6443/tcp | head -1 | sed 's/.*://')
kubectl config set-cluster k3d-anchor --server="https://127.0.0.1:${PORT}"
```

## GPU access, and what it costs

The NVIDIA device plugin **does not work here**. It needs NVML to enumerate
devices, and NVML returns `Not Supported` inside a pod nested in a k3d node
under WSL2. The same call succeeds one level up, in a plain
`docker run --gpus all`, so this is a nesting limitation rather than a WSL2-wide
one.

Pods therefore take the GPU directly: an init container copies the driver
libraries off the node into an `emptyDir` and the workload puts that on
`LD_LIBRARY_PATH`. Copying rather than bind-mounting the node's
`/usr/lib/x86_64-linux-gnu` matters, because that directory also holds the
node's glibc and shadowing the container's own breaks it.

**What this costs, stated plainly:** Kubernetes does not know a GPU exists.
`nvidia.com/gpu` is not an allocatable resource, so the scheduler cannot place
pods by GPU or prevent two of them contending for one card. On a single-GPU node
that is tolerable. On a real cluster it would not be, and the device plugin —
which works normally on bare-metal Linux — is the right answer there.

## Model choice is forced by 12 GB

Qwen2.5-7B in bf16 is roughly 15 GB and does not fit, so a quantization *delta*
cannot be measured at 7B on this card. The 3B pair fits both ways, which is what
makes a before-and-after comparison possible:

| Model | Precision | Weights |
|---|---|---|
| `Qwen/Qwen2.5-3B-Instruct` | bf16 | ~6.2 GB |
| `Qwen/Qwen2.5-3B-Instruct-AWQ` | 4-bit AWQ | ~2.3 GB |

## Apply

```bash
kubectl create namespace anchor
kubectl apply -f deploy/k3s/10-nvidia-libs.yaml
kubectl apply -f deploy/k3s/20-vllm.yaml
kubectl -n anchor rollout status deploy/vllm --timeout=30m
```
