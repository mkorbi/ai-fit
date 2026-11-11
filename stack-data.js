/* stack-data.js — content of the Engine Stack tab: layers of a self-hosted LLM platform around vLLM, which purpose needs
 * which layer, options with licences and links, and planning estimates of what each layer needs beside the accelerators.
 * Researched 2026-11-11; figures marked est. have no vendor sizing behind them. */
const STACK = {
 "checked": "2026-11-11",
 "resourceBasis": "Per-layer resources are totals for the whole reference cluster, for the components beside the accelerators. cpu = vCPU unless the note says physical cores, ramGB = GB, storageGB = usable GB. gpu: true means the component itself runs on or takes accelerator memory. need: required | recommended | optional | none. 'est.' marks numbers without a vendor figure behind them. Training assumes a 70B dense model (continued pre-training) so the three purposes share one model size.",
 "sources": {
  "kubeadm": "https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/",
  "etcdHardware": "https://etcd.io/docs/v3.6/op-guide/hardware/",
  "gpuOperatorPlatforms": "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/platform-support.html",
  "gpuOperatorValues": "https://github.com/NVIDIA/gpu-operator/blob/main/deployments/gpu-operator/values.yaml",
  "kserveValues": "https://github.com/kserve/kserve/blob/master/charts/kserve-resources/values.yaml",
  "vllmCpu": "https://docs.vllm.ai/en/latest/configuration/optimization.html",
  "prometheusStorage": "https://prometheus.io/docs/prometheus/latest/storage/",
  "rayHead": "https://docs.ray.io/en/latest/cluster/vms/user-guides/large-cluster-best-practices.html",
  "cephMinimum": "https://docs.ceph.com/en/latest/start/minimum-hardware/",
  "litellmProd": "https://docs.litellm.ai/docs/proxy/prod",
  "langfuseContainers": "https://langfuse.com/self-hosting/deployment/infrastructure/containers",
  "keycloakSizing": "https://www.keycloak.org/high-availability/multi-cluster/concepts-memory-and-cpu-sizing",
  "fabricManager": "https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/"
 },
 "purposes": {
  "inference": {
   "title": "Inference serving",
   "reference": "2 nodes × 8 GPUs, 70B model",
   "gpuMemoryRule": "Weights + KV cache + runtime overhead. Weights = params × bytes per param (70B BF16 = 140 GB, FP8 = 70 GB). KV per token = 2 × layers × KV heads × head dim × bytes (Llama-3-70B BF16: 2 × 80 × 8 × 128 × 2 B = 320 KB). One TP=8 replica per 8 × 80 GB node at gpu-memory-utilization 0.9: 576 GB − 140 GB weights − ~20 GB activations/CUDA graphs ≈ 410 GB, about 1.2M KV tokens per node.",
   "totals": {
    "controlNodes": "3 small Kubernetes control-plane nodes (4 vCPU, 16 GB, SSD) plus 3 storage nodes for the object store; or run both on an existing management cluster",
    "cpu": "~70 vCPU outside the GPU nodes; on each GPU node at least 2 + N = 10 physical cores for one vLLM TP=8 instance (vLLM docs), more in practice",
    "ramGB": "~200 GB outside the GPU nodes; 512 GB–1 TB host RAM per GPU node (page cache for 140 GB of weights, optional CPU KV offload) (est.)",
    "storage": "140 GB per 70B BF16 model version (70 GB FP8); 1–2 TB usable object store for 3–5 models and adapters; 1–2 TB local NVMe per GPU node as weight cache; metrics ~5 GB per 15 days",
    "network": "25–100 GbE front-end per node for API traffic; 100 GbE to storage loads 140 GB in ~15 s (10 GbE: ~2 min). No RDMA needed while each node runs its own TP=8 replica; add RDMA (200–400 Gb/s per GPU) only for one replica across both nodes or prefill/decode disaggregation with KV transfer",
    "note": "Two independent replicas behind a gateway are simpler and faster than one replica spanning both nodes for a 70B model."
   },
   "pitfalls": [
    "Host driver older than the CUDA version the vLLM image was built against: the container starts, then CUDA init fails. Pin driver branch and image together.",
    "Too few CPU cores per GPU node: vLLM needs 2 + N physical cores and the engine core busy-loops, so CPU limits or shared hyperthreads raise latency.",
    "Every pod pulls 140 GB from Hugging Face on start: mirror models into the object store and cache them on local NVMe, or scale-out takes minutes.",
    "Missing Fabric Manager (and NVLSM on HGX B200/B300) on NVSwitch systems: GPUs show up but multi-GPU jobs fail to initialise."
   ]
  },
  "finetune": {
   "title": "Fine-tuning",
   "reference": "1 node × 8 GPUs, LoRA or full fine-tune of a 70B model",
   "gpuMemoryRule": "LoRA: frozen weights (70B BF16 = 140 GB; QLoRA 4-bit ≈ 35–40 GB) + adapters with gradients and Adam state (rank 16 on all linear layers of Llama-3-70B ≈ 0.2B params × 16 B ≈ 3.3 GB) + activations (batch × sequence length; use gradient checkpointing). Full fine-tune: ~16 B/param (BF16 weights 2 + BF16 grads 2 + FP32 master 4 + Adam m and v 8) = 1.12 TB for 70B, more than 8 × 80 GB = 640 GB. Use 8 × B200 (1.44 TB), offload optimizer state to CPU (needs ~840 GB host RAM), or two nodes.",
   "totals": {
    "controlNodes": "3 small Kubernetes control-plane nodes or one Slurm controller; a single GPU node can also run jobs directly with torchrun",
    "cpu": "~30 vCPU outside the GPU node; 64+ cores on the GPU node for data loading and tokenisation (est.)",
    "ramGB": "~80 GB outside the GPU node; 1–2 TB host RAM on the GPU node when offloading optimizer state for full fine-tuning (70B × 12 B ≈ 840 GB)",
    "storage": "Base model 140 GB; LoRA adapters 0.4–3 GB each; full fine-tune checkpoint with optimizer state ≈ 14 B/param ≈ 1 TB, weights only 140 GB; datasets usually < 100 GB; plan 3–5 TB",
    "network": "NVLink inside the node does the heavy lifting; 25–100 GbE to storage. A 1 TB checkpoint at 1 GB/s takes ~17 min, so write to local NVMe first and upload in the background",
    "note": "LoRA of a 70B model fits 8 × 80 GB with room for long sequences; full fine-tuning does not without offload or bigger GPUs."
   },
   "pitfalls": [
    "Planning a full fine-tune of 70B on 8 × 80 GB: weights, gradients and Adam state alone are ~1.1 TB.",
    "Chat template or tokenizer differs between training and serving, so the tuned model looks worse in vLLM than in the notebook.",
    "Adapters trained on one base revision and served on another, or with a LoRA rank above vLLM's --max-lora-rank.",
    "No held-out evaluation: run lm-evaluation-harness and a task-specific set on base and tuned model before shipping."
   ]
  },
  "training": {
   "title": "Model training (pre-training, continued pre-training)",
   "reference": "32 nodes × 8 GPUs (256 GPUs), 70B dense model",
   "gpuMemoryRule": "16–20 B/param for weights, gradients and optimizer state (70B ≈ 1.1–1.4 TB), sharded across the data-parallel group by FSDP2 or ZeRO, + activations, which dominate at long sequences and depend on micro-batch, sequence length and recomputation. 256 × 80 GB = 20 TB, so model state is ~6% and the rest goes to activations and batch.",
   "totals": {
    "controlNodes": "3 Kubernetes control-plane nodes (8 vCPU, 32 GB, NVMe for etcd) or 2 Slurm controllers (slurmctld primary/backup) + 1 slurmdbd/MariaDB host, plus 2 login nodes; storage servers separate",
    "cpu": "~450 vCPU outside the GPU nodes (control plane, parallel file system servers, data pipeline); 32–64 cores per GPU node for data loading (est.)",
    "ramGB": "~1.8 TB outside the GPU nodes (est.); 1–2 TB host RAM per GPU node for checkpoint staging and data loaders",
    "storage": "Checkpoint ≈ 14 B/param ≈ 1 TB, keep 5–10 (5–10 TB); tokenised data at 4 B/token for vocabularies above 65k: 2T tokens ≈ 8 TB; raw text before filtering several times that (20–50 TB). Plan 50–100 TB parallel file system plus object storage for raw data.",
    "network": "Separate back-end fabric: 8 × 400 Gb/s RDMA NICs per node (InfiniBand NDR or RoCEv2), rail-optimised and non-blocking; front-end/storage network 100–200 Gb/s per node; ≥ 10 GB/s aggregate file system write so a 1 TB checkpoint lands in under 2 minutes",
    "note": "At 256 GPUs, node and GPU failures every few days are normal; budget for checkpoint frequency and automatic restarts."
   },
   "pitfalls": [
    "NCCL silently falls back to TCP sockets instead of InfiniBand/RoCE (no GPUDirect RDMA, wrong NCCL_IB_HCA, ACS enabled): check NCCL_DEBUG=INFO output on the first job.",
    "No failure handling: without frequent sharded checkpoints, health checks (DCGM diagnostics) and auto-restart, one bad node stalls 255 others.",
    "Storage sized for capacity, not throughput: checkpoint writes and data loading stall GPUs.",
    "Scheduler without gang scheduling: partial allocations deadlock multi-node jobs. Use Slurm, or Kueue/Volcano on Kubernetes."
   ]
  }
 },
 "layers": [
  {
   "id": "drivers",
   "name": "Drivers, libraries and fabric",
   "purpose": "Makes the accelerators and the RDMA network usable from containers: kernel driver, CUDA or ROCm user space, collective libraries and the NVSwitch fabric service.",
   "need": {
    "inference": "required",
    "finetune": "required",
    "training": "required"
   },
   "default": "NVIDIA data center driver + CUDA + NCCL + NVIDIA Container Toolkit (+ Fabric Manager on HGX)",
   "options": [
    {
     "name": "NVIDIA data center driver + CUDA",
     "url": "https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/",
     "license": "Proprietary (open GPU kernel modules MIT/GPL-2.0)",
     "oneLine": "Kernel driver and CUDA runtime; GPU Operator can install the driver as a container."
    },
    {
     "name": "AMD ROCm",
     "url": "https://rocm.docs.amd.com/",
     "license": "Mostly MIT / Apache-2.0",
     "oneLine": "AMD driver stack and HIP runtime for Instinct GPUs; vLLM ships ROCm images."
    },
    {
     "name": "NCCL",
     "url": "https://github.com/NVIDIA/nccl",
     "license": "Apache-2.0 (parts BSD-3-Clause)",
     "oneLine": "Collective communication over NVLink and InfiniBand/RoCE used by vLLM tensor parallelism and all training frameworks."
    },
    {
     "name": "RCCL",
     "url": "https://github.com/ROCm/rccl",
     "license": "BSD-3-Clause (with MIT parts)",
     "oneLine": "AMD's NCCL-compatible collective library."
    },
    {
     "name": "NVIDIA Fabric Manager",
     "url": "https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/",
     "license": "Proprietary",
     "oneLine": "Required on NVSwitch HGX/DGX systems (A100 to B300); B200/B300 also need NVLink Subnet Manager."
    },
    {
     "name": "NVIDIA Container Toolkit",
     "url": "https://github.com/NVIDIA/nvidia-container-toolkit",
     "license": "Apache-2.0",
     "oneLine": "Exposes GPUs to containers via CDI or the runtime hook."
    }
   ],
   "resources": {
    "inference": {
     "cpu": 2,
     "ramGB": 4,
     "storageGB": 60,
     "gpu": false,
     "note": "On each GPU node: ~1 core and 2 GB for driver daemons and Fabric Manager, ~30 GB for driver, CUDA and images (est.)."
    },
    "finetune": {
     "cpu": 1,
     "ramGB": 2,
     "storageGB": 30,
     "gpu": false,
     "note": "Same per-node footprint on the single GPU node (est.)."
    },
    "training": {
     "cpu": 32,
     "ramGB": 64,
     "storageGB": 960,
     "gpu": false,
     "note": "32 nodes × per-node footprint; add RDMA NIC drivers (DOCA/OFED) on every node (est.)."
    }
   }
  },
  {
   "id": "cluster",
   "name": "Cluster and scheduling",
   "purpose": "Places workloads on GPU nodes, advertises GPUs as schedulable resources and, for multi-node jobs, starts all ranks together (gang scheduling) with quotas between teams.",
   "need": {
    "inference": "required",
    "finetune": "recommended",
    "training": "required"
   },
   "default": "Kubernetes + NVIDIA GPU Operator + Kueue (Slurm for large training)",
   "options": [
    {
     "name": "Kubernetes",
     "url": "https://kubernetes.io/",
     "license": "Apache-2.0",
     "oneLine": "Container orchestration; kubeadm asks for 2+ CPUs and 2+ GB RAM per control-plane machine as a floor."
    },
    {
     "name": "NVIDIA GPU Operator",
     "url": "https://github.com/NVIDIA/gpu-operator",
     "license": "Apache-2.0",
     "oneLine": "Installs driver, container toolkit, device plugin, GPU feature discovery, DCGM exporter and MIG manager as DaemonSets; v26.7 supports Kubernetes 1.33–1.37."
    },
    {
     "name": "AMD GPU Operator",
     "url": "https://github.com/ROCm/gpu-operator",
     "license": "Apache-2.0",
     "oneLine": "Same role for AMD Instinct: driver, device plugin, node labeller, metrics exporter."
    },
    {
     "name": "NVIDIA Network Operator",
     "url": "https://github.com/Mellanox/network-operator",
     "license": "Apache-2.0",
     "oneLine": "RDMA drivers, SR-IOV and secondary networks for GPUDirect RDMA on Kubernetes."
    },
    {
     "name": "Slurm (+ Pyxis/enroot)",
     "url": "https://slurm.schedmd.com/",
     "license": "GPL-2.0-or-later",
     "oneLine": "HPC batch scheduler with native gang scheduling and topology-aware placement; Pyxis runs containers per job."
    },
    {
     "name": "Kueue",
     "url": "https://github.com/kubernetes-sigs/kueue",
     "license": "Apache-2.0",
     "oneLine": "Job queueing, quotas and all-or-nothing admission on Kubernetes."
    },
    {
     "name": "Volcano",
     "url": "https://github.com/volcano-sh/volcano",
     "license": "Apache-2.0",
     "oneLine": "Batch scheduler for Kubernetes with gang scheduling and network-topology-aware placement."
    },
    {
     "name": "KubeRay",
     "url": "https://github.com/ray-project/kuberay",
     "license": "Apache-2.0",
     "oneLine": "Operator for Ray clusters, jobs and Ray Serve on Kubernetes."
    }
   ],
   "resources": {
    "inference": {
     "cpu": 12,
     "ramGB": 48,
     "storageGB": 300,
     "gpu": false,
     "note": "3 control-plane nodes × 4 vCPU/16 GB (etcd's example for a 250-node cluster) with 100 GB SSD each; GPU Operator pod requests 200m/100Mi, limits 500m/350Mi, plus per-node DaemonSets ~0.5 core/1 GB (est.)."
    },
    "finetune": {
     "cpu": 6,
     "ramGB": 24,
     "storageGB": 200,
     "gpu": false,
     "note": "3 × 2 vCPU/8 GB control plane (etcd small example) or a single Slurm controller; skip entirely for a lone node run with torchrun."
    },
    "training": {
     "cpu": 40,
     "ramGB": 160,
     "storageGB": 1000,
     "gpu": false,
     "note": "3 control-plane nodes × 8 vCPU/32 GB plus Slurm controllers and slurmdbd/MariaDB (2 × 8 cores/16 GB); Slurm gives no hardware minimum, sizes are est."
    }
   }
  },
  {
   "id": "storage",
   "name": "Storage and model artifacts",
   "purpose": "Holds model weights, adapters, checkpoints and datasets close to the GPUs, and replaces live downloads from the Hugging Face Hub with a local mirror.",
   "need": {
    "inference": "required",
    "finetune": "required",
    "training": "required"
   },
   "default": "Ceph (RGW for S3) for objects + local NVMe weight cache; Lustre or Weka for training checkpoints",
   "options": [
    {
     "name": "Ceph (RADOS Gateway)",
     "url": "https://docs.ceph.com/en/latest/radosgw/",
     "license": "LGPL-2.1/LGPL-3.0",
     "oneLine": "Self-hosted S3 and block/file storage; per docs ≥ 4 GiB RAM and 4–6 threads per NVMe OSD, ≥ 5 GB and 2 cores per monitor."
    },
    {
     "name": "SeaweedFS",
     "url": "https://github.com/seaweedfs/seaweedfs",
     "license": "Apache-2.0",
     "oneLine": "Lightweight S3-compatible object store and filer for small clusters."
    },
    {
     "name": "MinIO",
     "url": "https://github.com/minio/minio",
     "license": "AGPL-3.0",
     "oneLine": "S3-compatible store; the community repository is archived and no longer maintained, new deployments go to the commercial AIStor editions."
    },
    {
     "name": "Lustre",
     "url": "https://www.lustre.org/",
     "license": "GPL-2.0",
     "oneLine": "Parallel file system for high-throughput checkpoint and dataset I/O on large training clusters."
    },
    {
     "name": "WEKA",
     "url": "https://www.weka.io/",
     "license": "Proprietary",
     "oneLine": "NVMe-based parallel file system with GPUDirect Storage support."
    },
    {
     "name": "JuiceFS",
     "url": "https://github.com/juicedata/juicefs",
     "license": "Apache-2.0",
     "oneLine": "POSIX file system on top of object storage with local caching, useful for sharing weights across nodes."
    },
    {
     "name": "Hugging Face Hub client (cache / mirror)",
     "url": "https://huggingface.co/docs/huggingface_hub/guides/manage-cache",
     "license": "Apache-2.0",
     "oneLine": "Download once into a shared cache or object store; point HF_ENDPOINT/HF_HUB_CACHE at the mirror."
    }
   ],
   "resources": {
    "inference": {
     "cpu": 24,
     "ramGB": 64,
     "storageGB": 2000,
     "gpu": false,
     "note": "3-node Ceph with 2 NVMe OSDs each (6 × 4 GiB + 3 monitors × 5 GB per Ceph minimums, rounded up) for 1–2 TB usable at 3× replication; plus 1–2 TB local NVMe per GPU node."
    },
    "finetune": {
     "cpu": 24,
     "ramGB": 64,
     "storageGB": 5000,
     "gpu": false,
     "note": "Base model 140 GB, 3 full checkpoints at ~1 TB each (14 B/param), adapters, datasets; local NVMe on the GPU node for checkpoint staging (est.)."
    },
    "training": {
     "cpu": 96,
     "ramGB": 768,
     "storageGB": 100000,
     "gpu": false,
     "note": "Parallel FS servers sized for ≥ 10 GB/s write (1 TB checkpoint in < 2 min) and dataloader reads; 50–100 TB covers 5–10 checkpoints plus tokenised and raw data (est.)."
    }
   }
  },
  {
   "id": "data",
   "name": "Data pipelines",
   "purpose": "Turns raw text into training-ready data: extraction, filtering, deduplication, PII removal, tokenisation and packing.",
   "need": {
    "inference": "none",
    "finetune": "recommended",
    "training": "required"
   },
   "default": "datatrove (on Slurm or local) or Ray Data on KubeRay",
   "options": [
    {
     "name": "datatrove",
     "url": "https://github.com/huggingface/datatrove",
     "license": "Apache-2.0",
     "oneLine": "Hugging Face's pipeline library for filtering, dedup and tokenisation at web scale; runs locally, on Slurm or Ray."
    },
    {
     "name": "Ray Data",
     "url": "https://docs.ray.io/en/latest/data/data.html",
     "license": "Apache-2.0",
     "oneLine": "Distributed data processing in Python, also used for batch inference with vLLM."
    },
    {
     "name": "Apache Spark",
     "url": "https://github.com/apache/spark",
     "license": "Apache-2.0",
     "oneLine": "General-purpose distributed processing when the data already lives in a lakehouse."
    },
    {
     "name": "NeMo Curator",
     "url": "https://github.com/NVIDIA-NeMo/Curator",
     "license": "Apache-2.0",
     "oneLine": "NVIDIA's curation toolkit with GPU-accelerated dedup and quality classifiers."
    }
   ],
   "resources": {
    "inference": {
     "cpu": 0,
     "ramGB": 0,
     "storageGB": 0,
     "gpu": false,
     "note": "Not needed for serving; batch inference jobs can reuse Ray Data."
    },
    "finetune": {
     "cpu": 16,
     "ramGB": 64,
     "storageGB": 200,
     "gpu": false,
     "note": "One CPU node or the GPU node's host CPUs for cleaning and tokenising < 100 GB of data (est.)."
    },
    "training": {
     "cpu": 256,
     "ramGB": 1024,
     "storageGB": 20000,
     "gpu": false,
     "note": "CPU pool for filtering, dedup and tokenising ~2T tokens; Ray head node 8 CPUs/32 GB per Ray docs; scratch space separate from the training FS; NeMo Curator can use GPUs for dedup (est.)."
    }
   }
  },
  {
   "id": "inference-engine",
   "name": "Inference engine",
   "purpose": "Loads the model onto the GPUs and runs batched generation with a paged KV cache behind an OpenAI-compatible API.",
   "need": {
    "inference": "required",
    "finetune": "optional",
    "training": "optional"
   },
   "default": "vLLM",
   "options": [
    {
     "name": "vLLM",
     "url": "https://github.com/vllm-project/vllm",
     "license": "Apache-2.0",
     "oneLine": "Reference engine: PagedAttention, continuous batching, prefix caching, TP/PP/EP, multi-LoRA, NVIDIA/AMD/TPU and more."
    },
    {
     "name": "SGLang",
     "url": "https://github.com/sgl-project/sglang",
     "license": "Apache-2.0",
     "oneLine": "Engine with RadixAttention prefix caching and strong large-MoE support."
    },
    {
     "name": "TensorRT-LLM",
     "url": "https://github.com/NVIDIA/TensorRT-LLM",
     "license": "Apache-2.0",
     "oneLine": "NVIDIA-only engine with kernels tuned for Hopper and Blackwell, FP8/FP4."
    },
    {
     "name": "llama.cpp",
     "url": "https://github.com/ggml-org/llama.cpp",
     "license": "MIT",
     "oneLine": "C/C++ engine for GGUF models on CPUs, single GPUs and edge devices."
    }
   ],
   "resources": {
    "inference": {
     "cpu": 20,
     "ramGB": 1024,
     "storageGB": 600,
     "gpu": true,
     "note": "Per vLLM docs at least 2 + N = 10 physical cores per TP=8 instance (2 nodes = 20 physical); 512 GB host RAM per node for weight page cache and CPU KV offload; ~300 GB local NVMe per node for weights and image (est.)."
    },
    "finetune": {
     "cpu": 10,
     "ramGB": 64,
     "storageGB": 0,
     "gpu": true,
     "note": "Optional: vLLM serves generations for online RL (TRL GRPO) or eval on the same GPUs."
    },
    "training": {
     "cpu": 10,
     "ramGB": 64,
     "storageGB": 0,
     "gpu": true,
     "note": "Optional: evaluation of checkpoints with lm-evaluation-harness on a vLLM backend, borrowing one node."
    }
   }
  },
  {
   "id": "training-framework",
   "name": "Distributed training framework",
   "purpose": "Shards weights, gradients and optimizer state across GPUs and nodes (data, tensor, pipeline, context and expert parallelism) and writes distributed checkpoints.",
   "need": {
    "inference": "none",
    "finetune": "required",
    "training": "required"
   },
   "default": "PyTorch FSDP2 via TorchTitan (Megatron-Core above ~1000 GPUs or for large MoE)",
   "options": [
    {
     "name": "PyTorch FSDP2",
     "url": "https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html",
     "license": "BSD-3-Clause",
     "oneLine": "Per-parameter sharding built into PyTorch; base of TorchTitan, TRL and Axolotl multi-GPU runs."
    },
    {
     "name": "TorchTitan",
     "url": "https://github.com/pytorch/torchtitan",
     "license": "BSD-3-Clause",
     "oneLine": "PyTorch-native pre-training platform combining FSDP2, TP, PP, CP, float8 and distributed checkpointing."
    },
    {
     "name": "Megatron-Core",
     "url": "https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core",
     "license": "BSD-3-Clause",
     "oneLine": "NVIDIA's parallelism library for the largest dense and MoE pre-training runs."
    },
    {
     "name": "DeepSpeed",
     "url": "https://github.com/deepspeedai/DeepSpeed",
     "license": "Apache-2.0",
     "oneLine": "ZeRO stages 1–3 and CPU/NVMe offload; common for full fine-tunes that exceed GPU memory."
    },
    {
     "name": "NeMo Megatron-Bridge",
     "url": "https://github.com/NVIDIA-NeMo/Megatron-Bridge",
     "license": "Apache-2.0",
     "oneLine": "NeMo's training library on Megatron-Core with Hugging Face checkpoint conversion both ways."
    }
   ],
   "resources": {
    "inference": {
     "cpu": 0,
     "ramGB": 0,
     "storageGB": 0,
     "gpu": false,
     "note": "Not used."
    },
    "finetune": {
     "cpu": 64,
     "ramGB": 1024,
     "storageGB": 2000,
     "gpu": true,
     "note": "On the GPU node: 4–8 data-loader cores per GPU; ~840 GB host RAM if optimizer state is offloaded for a full fine-tune; local NVMe for checkpoint staging (est.)."
    },
    "training": {
     "cpu": 1536,
     "ramGB": 32768,
     "storageGB": 96000,
     "gpu": true,
     "note": "On the 32 GPU nodes: ~48 cores, 1 TB RAM and 3 TB local NVMe each for data loaders and async checkpoint staging (est.)."
    }
   }
  },
  {
   "id": "finetune-framework",
   "name": "Fine-tuning toolkit",
   "purpose": "Recipes for supervised fine-tuning, preference tuning and RL on top of the training framework, including LoRA/QLoRA adapters and chat templates.",
   "need": {
    "inference": "none",
    "finetune": "required",
    "training": "optional"
   },
   "default": "TRL + PEFT (Axolotl as a config-driven wrapper)",
   "options": [
    {
     "name": "PEFT",
     "url": "https://github.com/huggingface/peft",
     "license": "Apache-2.0",
     "oneLine": "LoRA, QLoRA and other adapter methods for Hugging Face models; adapters load directly into vLLM."
    },
    {
     "name": "TRL",
     "url": "https://github.com/huggingface/trl",
     "license": "Apache-2.0",
     "oneLine": "SFT, DPO, GRPO and reward-model trainers; can use vLLM for generation during RL."
    },
    {
     "name": "Axolotl",
     "url": "https://github.com/axolotl-ai-cloud/axolotl",
     "license": "Apache-2.0",
     "oneLine": "YAML-configured fine-tuning over FSDP2/DeepSpeed, multi-node capable."
    },
    {
     "name": "Unsloth",
     "url": "https://github.com/unslothai/unsloth",
     "license": "Apache-2.0",
     "oneLine": "Memory-saving kernels for LoRA/QLoRA, strongest on single GPUs."
    },
    {
     "name": "LLaMA-Factory",
     "url": "https://github.com/hiyouga/LLaMA-Factory",
     "license": "Apache-2.0",
     "oneLine": "Fine-tuning for many model families with a web UI and CLI."
    },
    {
     "name": "torchtune",
     "url": "https://github.com/meta-pytorch/torchtune",
     "license": "BSD-3-Clause",
     "oneLine": "PyTorch-native post-training recipes; no longer actively maintained since 2025, avoid for new setups."
    }
   ],
   "resources": {
    "inference": {
     "cpu": 0,
     "ramGB": 0,
     "storageGB": 0,
     "gpu": false,
     "note": "Not used; vLLM serves the resulting adapters with --enable-lora."
    },
    "finetune": {
     "cpu": 8,
     "ramGB": 32,
     "storageGB": 50,
     "gpu": true,
     "note": "Runs inside the training processes; extra host cost is tokenisation and packing (est.)."
    },
    "training": {
     "cpu": 8,
     "ramGB": 32,
     "storageGB": 50,
     "gpu": true,
     "note": "Only for post-training after pre-training (est.)."
    }
   }
  },
  {
   "id": "serving",
   "name": "Serving control plane",
   "purpose": "Deploys and scales engine replicas, routes requests by KV-cache and load, and handles prefill/decode disaggregation and LoRA placement.",
   "need": {
    "inference": "recommended",
    "finetune": "none",
    "training": "none"
   },
   "default": "llm-d (KServe LLMInferenceService as the API)",
   "options": [
    {
     "name": "llm-d",
     "url": "https://github.com/llm-d/llm-d",
     "license": "Apache-2.0",
     "oneLine": "Kubernetes-native distributed vLLM serving with KV-aware routing via the Gateway API Inference Extension and disaggregated prefill/decode."
    },
    {
     "name": "KServe",
     "url": "https://github.com/kserve/kserve",
     "license": "Apache-2.0",
     "oneLine": "Model serving CRDs; LLMInferenceService (v0.21) deploys vLLM with prefix-aware routing and disaggregation."
    },
    {
     "name": "vLLM production-stack",
     "url": "https://github.com/vllm-project/production-stack",
     "license": "Apache-2.0",
     "oneLine": "Helm chart with a router, LMCache KV offload and dashboards around vLLM."
    },
    {
     "name": "NVIDIA Dynamo",
     "url": "https://github.com/ai-dynamo/dynamo",
     "license": "Apache-2.0",
     "oneLine": "Engine-agnostic (vLLM, SGLang, TRT-LLM) distributed serving with KV-aware routing and disaggregation."
    },
    {
     "name": "AIBrix",
     "url": "https://github.com/vllm-project/aibrix",
     "license": "Apache-2.0",
     "oneLine": "vLLM-project control plane with LoRA management, autoscaling and distributed KV cache."
    },
    {
     "name": "Ray Serve",
     "url": "https://docs.ray.io/en/latest/serve/index.html",
     "license": "Apache-2.0",
     "oneLine": "Python serving framework on Ray with LLM APIs on top of vLLM."
    }
   ],
   "resources": {
    "inference": {
     "cpu": 6,
     "ramGB": 12,
     "storageGB": 10,
     "gpu": false,
     "note": "KServe chart defaults: 1 CPU/2 GiB for the main controller, 100m–1 CPU and 100Mi–1Gi for others; endpoint picker/router 2 replicas × 2 vCPU/4 GB (est.)."
    },
    "finetune": {
     "cpu": 0,
     "ramGB": 0,
     "storageGB": 0,
     "gpu": false,
     "note": "Not used."
    },
    "training": {
     "cpu": 0,
     "ramGB": 0,
     "storageGB": 0,
     "gpu": false,
     "note": "Not used."
    }
   }
  },
  {
   "id": "gateway",
   "name": "API gateway",
   "purpose": "One OpenAI-compatible endpoint for clients with API keys, per-team quotas and rate limits, usage accounting and failover between models and providers.",
   "need": {
    "inference": "recommended",
    "finetune": "none",
    "training": "none"
   },
   "default": "LiteLLM proxy",
   "options": [
    {
     "name": "LiteLLM",
     "url": "https://github.com/BerriAI/litellm",
     "license": "MIT (enterprise/ directory separate)",
     "oneLine": "Proxy with virtual keys, budgets and routing; docs size 1 vCPU and 4 GiB per worker pod, Postgres required, Redis for more than one instance."
    },
    {
     "name": "Agent Router (formerly Envoy AI Gateway)",
     "url": "https://github.com/theagentrouter/agent-router",
     "license": "Apache-2.0",
     "oneLine": "Envoy Gateway-based AI and MCP traffic control plane with token-based rate limits; renamed and moved to theagentrouter/agent-router."
    },
    {
     "name": "Gateway API Inference Extension",
     "url": "https://github.com/kubernetes-sigs/gateway-api-inference-extension",
     "license": "Apache-2.0",
     "oneLine": "InferencePool and endpoint picker so any conformant Gateway routes by model, queue depth and KV-cache state."
    },
    {
     "name": "Kong AI Gateway",
     "url": "https://developer.konghq.com/ai-gateway/",
     "license": "Apache-2.0 (some AI plugins Enterprise)",
     "oneLine": "AI proxy plugins for Kong Gateway."
    },
    {
     "name": "agentgateway",
     "url": "https://github.com/agentgateway/agentgateway",
     "license": "Apache-2.0",
     "oneLine": "Linux Foundation project contributed by Solo.io: OpenAI-compatible LLM gateway with budgets and failover, plus MCP tool and agent-to-agent (A2A) traffic in one proxy."
    }
   ],
   "resources": {
    "inference": {
     "cpu": 5,
     "ramGB": 14,
     "storageGB": 50,
     "gpu": false,
     "note": "2 LiteLLM pods × 1 vCPU/4 GiB (LiteLLM docs) + Postgres 2 vCPU/4 GB + Redis 1 vCPU/2 GB (est.)."
    },
    "finetune": {
     "cpu": 0,
     "ramGB": 0,
     "storageGB": 0,
     "gpu": false,
     "note": "Not used."
    },
    "training": {
     "cpu": 0,
     "ramGB": 0,
     "storageGB": 0,
     "gpu": false,
     "note": "Not used."
    }
   }
  },
  {
   "id": "observability",
   "name": "Observability",
   "purpose": "GPU, node and engine metrics, logs and traces: utilisation, memory, XID errors, NVLink/IB counters, queue depth, TTFT and tokens per second, and LLM request traces.",
   "need": {
    "inference": "required",
    "finetune": "recommended",
    "training": "required"
   },
   "default": "Prometheus + DCGM exporter + Grafana (+ OpenTelemetry Collector; Langfuse for LLM traces)",
   "options": [
    {
     "name": "Prometheus",
     "url": "https://github.com/prometheus/prometheus",
     "license": "Apache-2.0",
     "oneLine": "Metrics store; docs give 1–2 bytes per sample for disk sizing. vLLM exposes /metrics natively."
    },
    {
     "name": "NVIDIA DCGM exporter",
     "url": "https://github.com/NVIDIA/dcgm-exporter",
     "license": "Apache-2.0",
     "oneLine": "GPU telemetry for Prometheus; deployed by the GPU Operator."
    },
    {
     "name": "AMD Device Metrics Exporter",
     "url": "https://github.com/ROCm/device-metrics-exporter",
     "license": "Apache-2.0",
     "oneLine": "Same role for AMD Instinct GPUs."
    },
    {
     "name": "Grafana",
     "url": "https://github.com/grafana/grafana",
     "license": "AGPL-3.0",
     "oneLine": "Dashboards and alerting; vLLM and DCGM ship example dashboards."
    },
    {
     "name": "OpenTelemetry Collector",
     "url": "https://github.com/open-telemetry/opentelemetry-collector",
     "license": "Apache-2.0",
     "oneLine": "Receives and forwards traces, metrics and logs; vLLM can emit OTLP traces."
    },
    {
     "name": "Langfuse",
     "url": "https://github.com/langfuse/langfuse",
     "license": "MIT (ee directories separate)",
     "oneLine": "LLM tracing, prompt management and evals; needs Postgres, ClickHouse, Redis and S3; docs ask ≥ 2 CPU/4 GB per container."
    }
   ],
   "resources": {
    "inference": {
     "cpu": 20,
     "ramGB": 56,
     "storageGB": 300,
     "gpu": false,
     "note": "Prometheus ~20k series at 15 s scrape × 2 B × 15 days ≈ 4 GB, 2 vCPU/8 GB; Grafana 1 vCPU/1 GB; DCGM exporter per node; Langfuse web × 2 + worker at 2 CPU/4 GB each plus ClickHouse/Postgres/Redis ~8 vCPU/32 GB (est.)."
    },
    "finetune": {
     "cpu": 4,
     "ramGB": 12,
     "storageGB": 50,
     "gpu": false,
     "note": "Prometheus, Grafana and DCGM exporter for one node (est.)."
    },
    "training": {
     "cpu": 12,
     "ramGB": 48,
     "storageGB": 1000,
     "gpu": false,
     "note": "~75k series (32 nodes + 256 GPUs) ≈ 13 GB per 15 days at 2 B/sample; logs from 256 ranks dominate storage; Thanos or similar for long retention (est.)."
    }
   }
  },
  {
   "id": "tracking",
   "name": "Experiment tracking, registry and evaluation",
   "purpose": "Records runs, hyperparameters and loss curves, versions models and adapters, and runs benchmark and task evals before a model is promoted to serving.",
   "need": {
    "inference": "recommended",
    "finetune": "required",
    "training": "required"
   },
   "default": "MLflow (tracking + model registry) + lm-evaluation-harness",
   "options": [
    {
     "name": "MLflow",
     "url": "https://github.com/mlflow/mlflow",
     "license": "Apache-2.0",
     "oneLine": "Tracking server and model registry; artifacts go to the object store, metadata to Postgres."
    },
    {
     "name": "Weights & Biases",
     "url": "https://github.com/wandb/wandb",
     "license": "MIT client, proprietary server",
     "oneLine": "Hosted or self-managed tracking; the self-managed server needs a license."
    },
    {
     "name": "Aim",
     "url": "https://github.com/aimhubio/aim",
     "license": "Apache-2.0",
     "oneLine": "Self-hosted experiment tracker focused on comparing many runs."
    },
    {
     "name": "lm-evaluation-harness",
     "url": "https://github.com/EleutherAI/lm-evaluation-harness",
     "license": "MIT",
     "oneLine": "Standard benchmark runner; can use vLLM or an OpenAI-compatible endpoint as backend."
    },
    {
     "name": "NeMo Evaluator",
     "url": "https://github.com/NVIDIA-NeMo/Evaluator",
     "license": "Apache-2.0",
     "oneLine": "Evaluation library and service for benchmarks against deployed endpoints."
    }
   ],
   "resources": {
    "inference": {
     "cpu": 2,
     "ramGB": 4,
     "storageGB": 20,
     "gpu": false,
     "note": "Registry and eval results only; eval jobs borrow serving GPUs (est.)."
    },
    "finetune": {
     "cpu": 4,
     "ramGB": 8,
     "storageGB": 100,
     "gpu": false,
     "note": "MLflow server + Postgres; adapters and checkpoints stored in the object store (est.)."
    },
    "training": {
     "cpu": 8,
     "ramGB": 16,
     "storageGB": 200,
     "gpu": false,
     "note": "Metrics from 256 ranks; log from rank 0 or aggregate to keep the server light (est.)."
    }
   }
  },
  {
   "id": "security",
   "name": "Identity, secrets and policy",
   "purpose": "Single sign-on for users and services, storage for tokens such as Hugging Face and registry credentials, and admission policies (approved images, GPU quotas, no privileged pods).",
   "need": {
    "inference": "required",
    "finetune": "recommended",
    "training": "recommended"
   },
   "default": "Keycloak (OIDC) + OpenBao + Kyverno",
   "options": [
    {
     "name": "Keycloak",
     "url": "https://github.com/keycloak/keycloak",
     "license": "Apache-2.0",
     "oneLine": "OIDC/SAML identity provider; docs give 1250 MB base RAM per pod and 1 vCPU per 15 password logins/s."
    },
    {
     "name": "OpenBao",
     "url": "https://github.com/openbao/openbao",
     "license": "MPL-2.0",
     "oneLine": "Open-source fork of Vault for secrets and dynamic credentials."
    },
    {
     "name": "HashiCorp Vault",
     "url": "https://github.com/hashicorp/vault",
     "license": "BUSL-1.1",
     "oneLine": "Secrets manager; source-available licence since 2023."
    },
    {
     "name": "Kyverno",
     "url": "https://github.com/kyverno/kyverno",
     "license": "Apache-2.0",
     "oneLine": "Kubernetes policies written as YAML."
    },
    {
     "name": "OPA Gatekeeper",
     "url": "https://github.com/open-policy-agent/gatekeeper",
     "license": "Apache-2.0",
     "oneLine": "Kubernetes admission policies written in Rego."
    }
   ],
   "resources": {
    "inference": {
     "cpu": 8,
     "ramGB": 16,
     "storageGB": 50,
     "gpu": false,
     "note": "Keycloak 2 pods × 2 vCPU/2 GB (1250 MB base + headroom) + Postgres; OpenBao 3-node Raft × 0.5 vCPU/1 GB; Kyverno 3 replicas (est.)."
    },
    "finetune": {
     "cpu": 4,
     "ramGB": 8,
     "storageGB": 20,
     "gpu": false,
     "note": "Secrets store and policies; SSO often shared with the serving cluster (est.)."
    },
    "training": {
     "cpu": 6,
     "ramGB": 12,
     "storageGB": 30,
     "gpu": false,
     "note": "Slurm clusters typically use LDAP/SSSD plus MUNGE instead of OIDC on the nodes (est.)."
    }
   }
  }
 ]
};
if (typeof module !== 'undefined') module.exports = STACK;
