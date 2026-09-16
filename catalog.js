/* catalog.js — hardware and model catalogs for Context Budget.
 *
 * Units
 *   mem      GB (decimal, 1e9 bytes) of accelerator memory
 *   bw       GB/s memory bandwidth
 *   tflops   dense TFLOPS (no sparsity) per precision; missing key = no native support
 *   linkBw   GB/s per accelerator for the intra-node fabric (NVLink, Infinity Fabric, ICI...)
 *   nodeGpus accelerators per node that share that fabric
 *   arch     chip generation key into ARCHS: decides native compute formats and loadable weight-only formats
 *   tdp      W per accelerator (null = not published)
 *   price    USD per accelerator-hour, approximate on-demand cloud rate (editable, null = unknown)
 *   approx   true when a spec is announced, derived or otherwise not vendor-confirmed
 *   source   where the numbers come from: url, kind (datasheet = vendor spec page, derived = vendor page plus arithmetic such as
 *            halving a sparsity figure, announcement = pre-release claims), date checked; tools/check-sources.js re-checks
 *   priceSource  cloud price page the $/h figure was read from (on-demand, rounded)
 */
const LINKS = {
  nvlink: { name: 'NVLink / NVSwitch', latencyUs: 15 },
  if:     { name: 'AMD Infinity Fabric', latencyUs: 20 },
  ici:    { name: 'TPU ICI', latencyUs: 20 },
  nlink:  { name: 'NeuronLink', latencyUs: 25 },
  roce:   { name: 'On-package RoCE (Gaudi)', latencyUs: 40 },
  pcie5:  { name: 'PCIe 5.0 x16 (no NVLink)', bw: 64, latencyUs: 40 },
  pcie4:  { name: 'PCIe 4.0 x16 (no NVLink)', bw: 32, latencyUs: 45 },
  none:   { name: 'Single device', bw: 0, latencyUs: 0 },
};

/* Inter-node network, gbps is per node (all NICs together). */
const NETWORKS = [
  { id: 'none',   name: 'None (single node only)',                      gbps: 0,    latencyUs: 0 },
  { id: 'eth10',  name: '10 GbE, TCP',                                  gbps: 10,   latencyUs: 250 },
  { id: 'eth25',  name: '25 GbE, TCP',                                  gbps: 25,   latencyUs: 200 },
  { id: 'eth100', name: '100 GbE, RoCE',                                gbps: 100,  latencyUs: 90 },
  { id: 'eth200', name: '200 GbE, RoCE',                                gbps: 200,  latencyUs: 80 },
  { id: 'ib400',  name: '400 Gb/s InfiniBand NDR or 400 GbE RoCE',      gbps: 400,  latencyUs: 60 },
  { id: 'ib800',  name: '800 Gb/s InfiniBand XDR or 800 GbE',           gbps: 800,  latencyUs: 50 },
  { id: 'ib3200', name: '8 × 400 Gb/s per node (3.2 Tb/s, DGX-class)',  gbps: 3200, latencyUs: 50 },
  { id: 'ib6400', name: '8 × 800 Gb/s per node (6.4 Tb/s)',             gbps: 6400, latencyUs: 45 },
];

/* Chip generations decide which number formats the tensor cores compute natively and which weight formats an
 * inference engine can still load by dequantizing to BF16 on the fly (weight-only: saves memory, not compute).
 * Anything else is not loadable. Sources: NVIDIA CUDA compute-capability tables and data sheets, AMD CDNA ISA
 * guides, Intel Gaudi 3 white paper, Google Cloud TPU docs, AWS Trainium2 docs, and the vLLM quantization
 * hardware-support table (weight-only paths). Keep this table in sync with those, not the per-device entries. */
const ARCHS = {
  sm80:   { name: 'NVIDIA Ampere (CC 8.0)',             native: ['bf16', 'int8'],               weightOnly: ['fp8', 'int4', 'fp4'], note: 'FP8, INT4 and FP4 weights run through Marlin dequant kernels' },
  sm86:   { name: 'NVIDIA Ampere (CC 8.6)',             native: ['bf16', 'int8'],               weightOnly: ['fp8', 'int4', 'fp4'], note: 'FP8, INT4 and FP4 weights run through Marlin dequant kernels' },
  sm89:   { name: 'NVIDIA Ada Lovelace (CC 8.9)',       native: ['bf16', 'fp8', 'int8'],        weightOnly: ['int4', 'fp4'] },
  sm90:   { name: 'NVIDIA Hopper (CC 9.0)',             native: ['bf16', 'fp8', 'int8'],        weightOnly: ['int4', 'fp4'] },
  sm100:  { name: 'NVIDIA Blackwell (CC 10.0)',         native: ['bf16', 'fp8', 'int8', 'fp4'], weightOnly: ['int4'] },
  sm103:  { name: 'NVIDIA Blackwell Ultra (CC 10.3)',   native: ['bf16', 'fp8', 'int8', 'fp4'], weightOnly: ['int4'] },
  sm120:  { name: 'NVIDIA Blackwell GB20x (CC 12.0)',   native: ['bf16', 'fp8', 'int8', 'fp4'], weightOnly: ['int4'] },
  sm121:  { name: 'NVIDIA GB10 (CC 12.1)',              native: ['bf16', 'fp8', 'int8', 'fp4'], weightOnly: ['int4'] },
  rubin:  { name: 'NVIDIA Rubin (announced)',           native: ['bf16', 'fp8', 'int8', 'fp4'], weightOnly: ['int4'], approx: true },
  gfx942: { name: 'AMD CDNA 3 (gfx942)',                native: ['bf16', 'fp8', 'int8'],        weightOnly: ['int4', 'fp4'], note: 'FP8 units use the FNUZ encoding; engines convert OCP FP8 checkpoints on load' },
  gfx950: { name: 'AMD CDNA 4 (gfx950)',                native: ['bf16', 'fp8', 'int8', 'fp4'], weightOnly: ['int4'] },
  cdna5:  { name: 'AMD CDNA 5 (announced)',             native: ['bf16', 'fp8', 'int8', 'fp4'], weightOnly: ['int4'], approx: true },
  gaudi3: { name: 'Intel Gaudi 3',                      native: ['bf16', 'fp8', 'int8'],        weightOnly: ['int4'] },
  tpuv5e: { name: 'Google TPU v5e',                     native: ['bf16', 'int8'],               weightOnly: ['int4'] },
  tpuv5p: { name: 'Google TPU v5p',                     native: ['bf16', 'int8'],               weightOnly: ['int4'] },
  tpuv6e: { name: 'Google TPU v6e (Trillium)',          native: ['bf16', 'int8'],               weightOnly: ['int4'] },
  tpuv7:  { name: 'Google TPU v7 (Ironwood)',           native: ['bf16', 'fp8', 'int8'],        weightOnly: ['int4'] },
  trn2:   { name: 'AWS Trainium2',                      native: ['bf16', 'fp8', 'int8'],        weightOnly: ['int4'] },
  apple:  { name: 'Apple GPU (Metal)',                  native: ['bf16'],                       weightOnly: ['int8', 'int4'], note: 'MLX and llama.cpp weight-only quantization; no FP8 or FP4 kernels' },
};

/* Inference-engine support, transcribed from each engine's own documentation on the date given (re-check when the
 * docs move on). Per weight format and hardware family: 'native' = kernels compute in that format,
 * 'weight-only' = weights stay quantized but matmuls run in BF16, absent = the engine cannot load it there.
 * kvCache lists the KV-cache dtypes the engine can run per family. The engine cannot exceed what ARCHS says the
 * silicon can do; the planner takes the weaker of the two. Families group the ARCHS keys. */
const ENGINE_FAMILY = { sm80: 'ampere', sm86: 'ampere', sm89: 'ada', sm90: 'hopper', sm100: 'blackwell', sm103: 'blackwell', sm120: 'blackwell-sm120', sm121: 'blackwell-sm120', rubin: 'blackwell', gfx942: 'cdna3', gfx950: 'cdna4', cdna5: 'cdna4', gaudi3: 'gaudi', tpuv5e: 'tpu', tpuv5p: 'tpu', tpuv6e: 'tpu', tpuv7: 'tpu', trn2: 'other', apple: 'other' };
const ENGINES = {
  none: { name: 'Hardware capability only', note: 'What the silicon can do, ignoring engine kernels. Pick an engine for what you can actually deploy.' },
  vllm: {
    name: 'vLLM', version: 'docs "latest"', checked: '2026-09-15',
    docs: { weights: 'https://docs.vllm.ai/en/latest/features/quantization/', kv: 'https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/' },
    weights: {
      fp8:  { ampere: 'weight-only', ada: 'native', hopper: 'native', blackwell: 'native', 'blackwell-sm120': 'native', cdna3: 'native', cdna4: 'native', gaudi: 'native' },
      int8: { ampere: 'native', ada: 'native', hopper: 'native', blackwell: 'native', 'blackwell-sm120': 'native' },
      int4: { ampere: 'weight-only', ada: 'weight-only', hopper: 'weight-only', blackwell: 'weight-only', 'blackwell-sm120': 'weight-only', gaudi: 'weight-only' },
      fp4:  { ampere: 'weight-only', ada: 'weight-only', hopper: 'weight-only', blackwell: 'native', 'blackwell-sm120': 'native', cdna4: 'native' },
    },
    kvCache: { fp8: ['ampere', 'ada', 'hopper', 'blackwell', 'blackwell-sm120', 'cdna3', 'cdna4', 'gaudi'] },
    notes: [
      'Support table (AWQ, GPTQ, Marlin, llm-compressor INT8/FP8, bitsandbytes, GGUF) covers Volta through Hopper, AMD, Intel and CPU; Blackwell inherits the Hopper column plus native NVFP4 (ModelOpt / compressed-tensors NVFP4).',
      'FP8 W8A8 needs Ada or newer or AMD; on Ampere FP8 and FP4 checkpoints run through Marlin as weight-only.',
      'AWQ and GPTQ are listed as unsupported on AMD GPUs; INT8 W8A8 is NVIDIA and CPU only.',
      'KV cache: kv_cache_dtype fp8_e4m3 on CUDA and ROCm, fp8_e5m2 on CUDA; per-head scales need FlashAttention plus llm-compressor calibration; sliding-window layers are sensitive (kv-cache-dtype-skip-layers). No INT8 or INT4 KV cache.',
      'Gaudi quantization lives in vLLM-Gaudi; TPU support is documented separately (TPU-Inference).',
      'MXFP4 on CDNA 4 comes from the Quark / AITER path, not from the general table.',
    ],
  },
  sglang: {
    name: 'SGLang', version: 'docs "latest"', checked: '2026-09-15',
    docs: { weights: 'https://docs.sglang.ai/advanced_features/quantization.html', kv: 'https://docs.sglang.ai/advanced_features/server_arguments.html' },
    weights: {
      fp8:  { ampere: 'weight-only', ada: 'native', hopper: 'native', blackwell: 'native', 'blackwell-sm120': 'native', cdna3: 'native', cdna4: 'native' },
      int8: { ampere: 'native', ada: 'native', hopper: 'native', blackwell: 'native', 'blackwell-sm120': 'native' },
      int4: { ampere: 'weight-only', ada: 'weight-only', hopper: 'weight-only', blackwell: 'weight-only', 'blackwell-sm120': 'weight-only' },
      fp4:  { ampere: 'weight-only', ada: 'weight-only', hopper: 'weight-only', blackwell: 'native', 'blackwell-sm120': 'native', cdna3: 'weight-only', cdna4: 'native' },
    },
    kvCache: { fp8: ['ampere', 'ada', 'hopper', 'blackwell', 'blackwell-sm120', 'cdna3', 'cdna4'] },
    notes: [
      'Method table: compressed-tensors on NVIDIA and AMD (Aiter FP8/MoE paths on AMD); awq_marlin and gptq_marlin are CUDA-only (plain gptq removed on NVIDIA and AMD); modelopt FP8 needs Hopper or newer.',
      'modelopt_fp4: native FP4 on SM100 and newer (flashinfer backends, SM120 via flashinfer_cutlass), Marlin W4A16 fallback on SM80 to SM90; petit_nvfp4 brings NVFP4 to MI250/MI300X/MI325X as weight-only; quark_mxfp4 runs MXFP4 natively on CDNA 4 (gfx95x).',
      'KV cache: --kv-cache-dtype fp8_e5m2 or fp8_e4m3 (server arguments page, not re-verified from the quantization page).',
    ],
  },
  trtllm: {
    name: 'TensorRT-LLM', version: 'docs "latest" (PyTorch backend)', checked: '2026-09-15',
    docs: { weights: 'https://nvidia.github.io/TensorRT-LLM/features/quantization.html', kv: 'https://nvidia.github.io/TensorRT-LLM/features/quantization.html' },
    weights: {
      fp8:  { ada: 'native', hopper: 'native', blackwell: 'native', 'blackwell-sm120': 'native' },
      int8: {},
      int4: { ampere: 'weight-only', ada: 'weight-only', hopper: 'weight-only', blackwell: 'weight-only' },
      fp4:  { blackwell: 'native', 'blackwell-sm120': 'native' },
    },
    kvCache: { fp8: ['ampere', 'ada', 'hopper', 'blackwell', 'blackwell-sm120'] },
    notes: [
      'Hardware table: NVFP4 and MXFP4 on Blackwell (sm100/103 and sm120); FP8 per-tensor on Ada, Hopper and Blackwell, FP8 block scaling on Hopper and sm100/103, rowwise on Hopper only; W4A16 AWQ/GPTQ on Ampere through sm100/103 (W4A8 from Ada up) but not on sm120; FP8 KV cache on every generation from Ampere; NVFP4 KV cache on sm100/103 (not modeled here).',
      'INT8 (SmoothQuant) does not appear in the current PyTorch-backend matrix, so it is treated as unavailable. NVIDIA GPUs only. The same page also lists per-model-family support (e.g. NVFP4 for LLaMA 4, Mixtral, Qwen 3 and DeepSeek-R1), which this planner does not encode.',
    ],
  },
};

const HARDWARE = [
  // NVIDIA data center
  { id: 'a100-sxm-80', arch: 'sm80',  vendor: 'NVIDIA', name: 'A100 SXM 80 GB',                gen: 'Ampere',        mem: 80,  bw: 2039,  tflops: { fp16: 312,  int8: 624 },                          link: 'nvlink', linkBw: 600,  nodeGpus: 8,  tdp: 400,  price: 1.5, source: { url: 'https://www.nvidia.com/en-us/data-center/a100/', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://lambda.ai/service/gpu-cloud' },
  { id: 'a100-pcie-80', arch: 'sm80', vendor: 'NVIDIA', name: 'A100 PCIe 80 GB',               gen: 'Ampere',        mem: 80,  bw: 1935,  tflops: { fp16: 312,  int8: 624 },                          link: 'pcie4',  linkBw: 32,   nodeGpus: 8,  tdp: 300,  price: 1.3, source: { url: 'https://www.nvidia.com/en-us/data-center/a100/', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'a10g', arch: 'sm86',         vendor: 'NVIDIA', name: 'A10G 24 GB',                    gen: 'Ampere',        mem: 24,  bw: 600,   tflops: { fp16: 70,   int8: 140 },                          link: 'pcie4',  linkBw: 32,   nodeGpus: 8,  tdp: 150,  price: 0.9, approx: true, source: { url: 'https://aws.amazon.com/ec2/instance-types/g5/', kind: 'derived', date: '2026-09-15' }, priceSource: 'https://aws.amazon.com/ec2/instance-types/g5/' },
  { id: 'l4', arch: 'sm89',           vendor: 'NVIDIA', name: 'L4 24 GB',                      gen: 'Ada',           mem: 24,  bw: 300,   tflops: { fp16: 121,  fp8: 242,  int8: 242 },               link: 'pcie4',  linkBw: 32,   nodeGpus: 8,  tdp: 72,   price: 0.6, source: { url: 'https://www.nvidia.com/en-us/data-center/l4/', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'l40s', arch: 'sm89',         vendor: 'NVIDIA', name: 'L40S 48 GB',                    gen: 'Ada',           mem: 48,  bw: 864,   tflops: { fp16: 362,  fp8: 733,  int8: 733 },               link: 'pcie4',  linkBw: 32,   nodeGpus: 8,  tdp: 350,  price: 1.0, source: { url: 'https://www.nvidia.com/en-us/data-center/l40s/', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'h100-pcie', arch: 'sm90',    vendor: 'NVIDIA', name: 'H100 PCIe 80 GB',               gen: 'Hopper',        mem: 80,  bw: 2039,  tflops: { fp16: 756,  fp8: 1513, int8: 1513 },              link: 'pcie5',  linkBw: 64,   nodeGpus: 8,  tdp: 350,  price: 2.3, source: { url: 'https://www.nvidia.com/en-us/data-center/h100/', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'h100-nvl', arch: 'sm90',     vendor: 'NVIDIA', name: 'H100 NVL 94 GB',                gen: 'Hopper',        mem: 94,  bw: 3938,  tflops: { fp16: 835,  fp8: 1671, int8: 1671 },              link: 'pcie5',  linkBw: 64,   nodeGpus: 8,  tdp: 400,  price: 2.6, source: { url: 'https://www.nvidia.com/en-us/data-center/h100/', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'h100-sxm', arch: 'sm90',     vendor: 'NVIDIA', name: 'H100 SXM 80 GB',                gen: 'Hopper',        mem: 80,  bw: 3352,  tflops: { fp16: 989,  fp8: 1979, int8: 1979 },              link: 'nvlink', linkBw: 900,  nodeGpus: 8,  tdp: 700,  price: 2.9, source: { url: 'https://www.nvidia.com/en-us/data-center/h100/', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://lambda.ai/service/gpu-cloud' },
  { id: 'h200-sxm', arch: 'sm90',     vendor: 'NVIDIA', name: 'H200 SXM 141 GB',               gen: 'Hopper',        mem: 141, bw: 4800,  tflops: { fp16: 989,  fp8: 1979, int8: 1979 },              link: 'nvlink', linkBw: 900,  nodeGpus: 8,  tdp: 700,  price: 3.7, source: { url: 'https://www.nvidia.com/en-us/data-center/h200/', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://lambda.ai/service/gpu-cloud' },
  { id: 'h200-nvl', arch: 'sm90',     vendor: 'NVIDIA', name: 'H200 NVL 141 GB',               gen: 'Hopper',        mem: 141, bw: 4800,  tflops: { fp16: 835,  fp8: 1671, int8: 1671 },              link: 'pcie5',  linkBw: 64,   nodeGpus: 8,  tdp: 600,  price: 3.3, source: { url: 'https://www.nvidia.com/en-us/data-center/h200/', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'b200', arch: 'sm100',         vendor: 'NVIDIA', name: 'B200 (HGX) 180 GB',             gen: 'Blackwell',     mem: 180, bw: 8000,  tflops: { fp16: 2250, fp8: 4500, fp4: 9000,  int8: 4500 },  link: 'nvlink', linkBw: 1800, nodeGpus: 8,  tdp: 1000, price: 5.5, source: { url: 'https://www.nvidia.com/en-us/data-center/hgx/', kind: 'derived', date: '2026-09-15' }, priceSource: 'https://lambda.ai/service/gpu-cloud' },
  { id: 'gb200', arch: 'sm100',        vendor: 'NVIDIA', name: 'GB200 NVL72, per GPU, 186 GB',  gen: 'Blackwell',     mem: 186, bw: 8000,  tflops: { fp16: 2500, fp8: 5000, fp4: 10000, int8: 5000 },  link: 'nvlink', linkBw: 1800, nodeGpus: 72, tdp: 1200, price: 7.0, approx: true, source: { url: 'https://www.nvidia.com/en-us/data-center/gb200-nvl72/', kind: 'derived', date: '2026-09-15' }, priceSource: 'https://lambda.ai/service/gpu-cloud' },
  { id: 'b300', arch: 'sm103',         vendor: 'NVIDIA', name: 'B300 (HGX) 288 GB',             gen: 'Blackwell Ultra', mem: 288, bw: 8000, tflops: { fp16: 2250, fp8: 4500, fp4: 13500, int8: 4500 }, link: 'nvlink', linkBw: 1800, nodeGpus: 8,  tdp: 1100, price: 6.5, approx: true, source: { url: 'https://www.nvidia.com/en-us/data-center/hgx/', kind: 'derived', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'gb300', arch: 'sm103',        vendor: 'NVIDIA', name: 'GB300 NVL72, per GPU, 288 GB',  gen: 'Blackwell Ultra', mem: 288, bw: 8000, tflops: { fp16: 2500, fp8: 5000, fp4: 15000, int8: 5000 }, link: 'nvlink', linkBw: 1800, nodeGpus: 72, tdp: 1400, price: 8.5, approx: true, source: { url: 'https://www.nvidia.com/en-us/data-center/gb300-nvl72/', kind: 'derived', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'rubin', arch: 'rubin',        vendor: 'NVIDIA', name: 'Rubin VR200 (announced) 288 GB', gen: 'Rubin',        mem: 288, bw: 20000, tflops: { fp16: 8300, fp8: 16700, fp4: 50000, int8: 16700 }, link: 'nvlink', linkBw: 3600, nodeGpus: 72, tdp: 1800, price: null, approx: true, source: { url: 'https://nvidianews.nvidia.com/news/nvidia-vera-rubin-platform', kind: 'announcement', date: '2026-09-15' } },
  { id: 'rtx-pro-6000', arch: 'sm120', vendor: 'NVIDIA', name: 'RTX PRO 6000 Blackwell 96 GB',  gen: 'Blackwell (GB202)', mem: 96, bw: 1600, tflops: { fp16: 500, fp8: 1000, fp4: 2000, int8: 1000 }, link: 'pcie5', linkBw: 64, nodeGpus: 8, tdp: 600, price: 1.8, approx: true, source: { url: 'https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/', kind: 'derived', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'rtx-5090', arch: 'sm120',     vendor: 'NVIDIA', name: 'GeForce RTX 5090 32 GB',        gen: 'Blackwell (GB202)', mem: 32, bw: 1792, tflops: { fp16: 210, fp8: 420, fp4: 840, int8: 420 },   link: 'pcie5', linkBw: 64, nodeGpus: 8, tdp: 575, price: 0.7, approx: true, source: { url: 'https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/', kind: 'derived', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'rtx-4090', arch: 'sm89',     vendor: 'NVIDIA', name: 'GeForce RTX 4090 24 GB',        gen: 'Ada',           mem: 24,  bw: 1008,  tflops: { fp16: 165,  fp8: 330,  int8: 330 },               link: 'pcie4',  linkBw: 32,   nodeGpus: 8,  tdp: 450,  price: 0.4, source: { url: 'https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/', kind: 'derived', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'dgx-spark', arch: 'sm121',    vendor: 'NVIDIA', name: 'DGX Spark (GB10) 128 GB unified', gen: 'Blackwell',   mem: 128, bw: 273,   tflops: { fp16: 125,  fp8: 250,  fp4: 500,   int8: 250 },   link: 'none',   linkBw: 0,    nodeGpus: 1,  tdp: 140,  price: null, approx: true, source: { url: 'https://www.nvidia.com/en-us/products/workstations/dgx-spark/', kind: 'derived', date: '2026-09-15' } },
  // AMD
  { id: 'mi300x', arch: 'gfx942',       vendor: 'AMD',    name: 'Instinct MI300X 192 GB',        gen: 'CDNA 3',        mem: 192, bw: 5300,  tflops: { fp16: 1307, fp8: 2615, int8: 2615 },              link: 'if',     linkBw: 896,  nodeGpus: 8,  tdp: 750,  price: 2.4, source: { url: 'https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'mi325x', arch: 'gfx942',       vendor: 'AMD',    name: 'Instinct MI325X 256 GB',        gen: 'CDNA 3',        mem: 256, bw: 6000,  tflops: { fp16: 1307, fp8: 2615, int8: 2615 },              link: 'if',     linkBw: 896,  nodeGpus: 8,  tdp: 1000, price: 3.0, source: { url: 'https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'mi350x', arch: 'gfx950',       vendor: 'AMD',    name: 'Instinct MI350X 288 GB',        gen: 'CDNA 4',        mem: 288, bw: 8000,  tflops: { fp16: 2300, fp8: 4600, fp4: 9200,  int8: 4600 },  link: 'if',     linkBw: 1075, nodeGpus: 8,  tdp: 1000, price: 4.0, approx: true, source: { url: 'https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'mi355x', arch: 'gfx950',       vendor: 'AMD',    name: 'Instinct MI355X 288 GB',        gen: 'CDNA 4',        mem: 288, bw: 8000,  tflops: { fp16: 2500, fp8: 5000, fp4: 10000, int8: 5000 },  link: 'if',     linkBw: 1075, nodeGpus: 8,  tdp: 1400, price: 4.5, approx: true, source: { url: 'https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  { id: 'mi400', arch: 'cdna5',        vendor: 'AMD',    name: 'Instinct MI400 (announced) 432 GB', gen: 'CDNA 5',    mem: 432, bw: 19600, tflops: { fp16: 10000, fp8: 20000, fp4: 40000, int8: 20000 }, link: 'if',   linkBw: 2400, nodeGpus: 72, tdp: 1600, price: null, approx: true, source: { url: 'https://www.amd.com/en/products/accelerators/instinct/mi400.html', kind: 'announcement', date: '2026-09-15' } },
  // Intel
  { id: 'gaudi3', arch: 'gaudi3',       vendor: 'Intel',  name: 'Gaudi 3 128 GB',                gen: 'Gaudi 3',       mem: 128, bw: 3700,  tflops: { fp16: 1835, fp8: 1835, int8: 1835 },              link: 'roce',   linkBw: 525,  nodeGpus: 8,  tdp: 900,  price: 2.0, approx: true, source: { url: 'https://www.intel.com/content/www/us/en/products/details/processors/ai-accelerators/gaudi3.html', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://www.runpod.io/pricing' },
  // Google TPU (cloud only)
  { id: 'tpu-v5e', arch: 'tpuv5e',      vendor: 'Google', name: 'TPU v5e 16 GB',                 gen: 'TPU v5e',       mem: 16,  bw: 819,   tflops: { fp16: 197,  int8: 394 },                          link: 'ici',    linkBw: 200,  nodeGpus: 256, tdp: null, price: 1.2, source: { url: 'https://cloud.google.com/tpu/docs/v5e', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://cloud.google.com/tpu/pricing' },
  { id: 'tpu-v5p', arch: 'tpuv5p',      vendor: 'Google', name: 'TPU v5p 95 GB',                 gen: 'TPU v5p',       mem: 95,  bw: 2765,  tflops: { fp16: 459,  int8: 918 },                          link: 'ici',    linkBw: 600,  nodeGpus: 256, tdp: null, price: 4.2, source: { url: 'https://cloud.google.com/tpu/docs/v5p', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://cloud.google.com/tpu/pricing' },
  { id: 'tpu-v6e', arch: 'tpuv6e',      vendor: 'Google', name: 'TPU v6e Trillium 32 GB',        gen: 'TPU v6e',       mem: 32,  bw: 1640,  tflops: { fp16: 918,  int8: 1836 },                         link: 'ici',    linkBw: 400,  nodeGpus: 256, tdp: null, price: 2.7, approx: true, source: { url: 'https://cloud.google.com/tpu/docs/v6e', kind: 'datasheet', date: '2026-09-15' }, priceSource: 'https://cloud.google.com/tpu/pricing' },
  { id: 'tpu-v7', arch: 'tpuv7',       vendor: 'Google', name: 'TPU v7 Ironwood 192 GB',        gen: 'TPU v7',        mem: 192, bw: 7370,  tflops: { fp16: 2307, fp8: 4614, int8: 4614 },              link: 'ici',    linkBw: 1200, nodeGpus: 256, tdp: null, price: null, approx: true, source: { url: 'https://cloud.google.com/tpu/docs/tpu7x', kind: 'datasheet', date: '2026-09-15' } },
  // AWS
  { id: 'trn2', arch: 'trn2',         vendor: 'AWS',    name: 'Trainium2 96 GB',               gen: 'Trn2',          mem: 96,  bw: 2900,  tflops: { fp16: 667,  fp8: 1333, int8: 1333 },              link: 'nlink',  linkBw: 640,  nodeGpus: 16, tdp: 500,  price: null, approx: true, source: { url: 'https://aws.amazon.com/ai/machine-learning/trainium/', kind: 'derived', date: '2026-09-15' } },
  // Apple unified memory (single box, local use)
  { id: 'm3-ultra-512', arch: 'apple', vendor: 'Apple',  name: 'Mac Studio M3 Ultra 512 GB',    gen: 'M3 Ultra',      mem: 512, bw: 819,   tflops: { fp16: 30 },                                       link: 'none',   linkBw: 0,    nodeGpus: 1,  tdp: 300,  price: null, approx: true, source: { url: 'https://www.apple.com/mac-studio/specs/', kind: 'derived', date: '2026-09-15' } },
  { id: 'm4-max-128', arch: 'apple',   vendor: 'Apple',  name: 'M4 Max 128 GB',                 gen: 'M4 Max',        mem: 128, bw: 546,   tflops: { fp16: 18 },                                       link: 'none',   linkBw: 0,    nodeGpus: 1,  tdp: 140,  price: null, approx: true, source: { url: 'https://www.apple.com/macbook-pro/specs/', kind: 'derived', date: '2026-09-15' } },
];

/* Models.
 *   params / active   billions of parameters (total / activated per token)
 *   layers, dModel, nHeads, nKv, dHead   transformer shape; nKv is the number of KV heads (GQA)
 *   attn      layer groups: { n, type: 'full' | 'swa' | 'chunked' | 'mla' | 'linear', window, dc, dr }
 *             swa/chunked cap the KV cache of those layers at `window` tokens
 *             mla stores one latent vector of dc + dr per token (DeepSeek-style)
 *             linear layers (Gated DeltaNet, lightning attention, Mamba) keep a constant state per sequence
 *   moe       { experts, active } routed experts total / per token
 *   sparse    { topk } attention only over the top-k tokens (DeepSeek Sparse Attention)
 *   statePerSeqMB  constant per-sequence state of linear layers
 *   maxCtx    largest context the released weights support (with RoPE scaling if nativeCtx is lower)
 *   nativePrec  precision the checkpoint ships in (from the Hugging Face safetensors metadata)
 *   hf / hfMirror  Hugging Face repo the entry is verified against (tools/import-model.js --check); the mirror is an ungated copy
 *   variants  known checkpoints per weight format, found with --find-variants and re-verified with --check-variants
 *             (same shape as the base, quantization_config says what the name claims); org tells you who made it
 */
const MODELS = [
  { id: 'llama-3.2-3b', hf: 'meta-llama/Llama-3.2-3B-Instruct', hfMirror: 'unsloth/Llama-3.2-3B-Instruct',     family: 'Meta Llama',  name: 'Llama 3.2 3B',                        params: 3.2,   layers: 28,  dModel: 3072,  nHeads: 24,  nKv: 8,  dHead: 128, maxCtx: 131072, nativePrec: 'bf16', variants: { fp8: 'RedHatAI/Llama-3.2-3B-Instruct-FP8-dynamic', int8: 'RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8', int4: 'casperhansen/llama-3.2-3b-instruct-awq' } },
  { id: 'llama-3.1-8b', hf: 'meta-llama/Llama-3.1-8B-Instruct', hfMirror: 'unsloth/Llama-3.1-8B-Instruct',     family: 'Meta Llama',  name: 'Llama 3.1 8B',                        params: 8.0,   layers: 32,  dModel: 4096,  nHeads: 32,  nKv: 8,  dHead: 128, maxCtx: 131072, nativePrec: 'bf16', variants: { fp8: 'RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8-dynamic', int8: 'RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8', int4: 'hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4', fp4: 'nvidia/Llama-3.1-8B-Instruct-NVFP4' } },
  { id: 'llama-3.3-70b', hf: 'meta-llama/Llama-3.3-70B-Instruct', hfMirror: 'unsloth/Llama-3.3-70B-Instruct',    family: 'Meta Llama',  name: 'Llama 3.3 / 3.1 70B',                 params: 70.6,  layers: 80,  dModel: 8192,  nHeads: 64,  nKv: 8,  dHead: 128, maxCtx: 131072, nativePrec: 'bf16', variants: { fp8: 'RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic', int8: 'RedHatAI/Llama-3.3-70B-Instruct-quantized.w8a8', int4: 'RedHatAI/Llama-3.3-70B-Instruct-quantized.w4a16', fp4: 'nvidia/Llama-3.3-70B-Instruct-NVFP4' } },
  { id: 'llama-3.1-405b', hf: 'meta-llama/Llama-3.1-405B-Instruct', hfMirror: 'RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic',   family: 'Meta Llama',  name: 'Llama 3.1 405B',                      params: 405.9, layers: 126, dModel: 16384, nHeads: 128, nKv: 8,  dHead: 128, maxCtx: 131072, nativePrec: 'bf16', variants: { fp8: 'RedHatAI/Meta-Llama-3.1-405B-Instruct-FP8-dynamic', int8: 'RedHatAI/Meta-Llama-3.1-405B-Instruct-quantized.w8a8', int4: 'hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4', fp4: 'nvidia/Llama-3.1-405B-Instruct-NVFP4' } },
  { id: 'llama-4-scout', hf: 'meta-llama/Llama-4-Scout-17B-16E-Instruct', hfMirror: 'unsloth/Llama-4-Scout-17B-16E-Instruct',    family: 'Meta Llama',  name: 'Llama 4 Scout 17B-16E',               params: 109,   active: 17, layers: 48, dModel: 5120, nHeads: 40, nKv: 8, dHead: 128, attn: [{ n: 36, type: 'chunked', window: 8192 }, { n: 12, type: 'full' }], moe: { experts: 16, active: 1 }, maxCtx: 10485760, note: 'iRoPE: 3 of 4 layers use 8k chunked attention', nativePrec: 'bf16', variants: { fp8: 'RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic', int4: 'RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16', fp4: 'RedHatAI/Llama-4-Scout-17B-16E-Instruct-NVFP4' } },
  { id: 'llama-4-maverick', hf: 'meta-llama/Llama-4-Maverick-17B-128E-Instruct', hfMirror: 'unsloth/Llama-4-Maverick-17B-128E-Instruct', family: 'Meta Llama',  name: 'Llama 4 Maverick 17B-128E',           params: 400,   active: 17, layers: 48, dModel: 5120, nHeads: 40, nKv: 8, dHead: 128, attn: [{ n: 36, type: 'chunked', window: 8192 }, { n: 12, type: 'full' }], moe: { experts: 128, active: 1 }, maxCtx: 1048576, note: 'iRoPE: 3 of 4 layers use 8k chunked attention', nativePrec: 'bf16', variants: { fp8: 'RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8', int4: 'RedHatAI/Llama-4-Maverick-17B-128E-Instruct-quantized.w4a16', fp4: 'RedHatAI/Llama-4-Maverick-17B-128E-Instruct-NVFP4' } },
  { id: 'mixtral-8x7b', hf: 'mistralai/Mixtral-8x7B-Instruct-v0.1',     family: 'Mistral',     name: 'Mixtral 8x7B',                        params: 46.7,  active: 12.9, layers: 32, dModel: 4096, nHeads: 32, nKv: 8, dHead: 128, moe: { experts: 8, active: 2 }, maxCtx: 32768, nativePrec: 'bf16', variants: { fp8: 'RedHatAI/Mixtral-8x7B-Instruct-v0.1-FP8', int4: 'hugging-quants/Mixtral-8x7B-Instruct-v0.1-AWQ-INT4' } },
  { id: 'mixtral-8x22b', hf: 'mistralai/Mixtral-8x22B-Instruct-v0.1',    family: 'Mistral',     name: 'Mixtral 8x22B',                       params: 141,   active: 39,  layers: 56, dModel: 6144, nHeads: 48, nKv: 8, dHead: 128, moe: { experts: 8, active: 2 }, maxCtx: 65536, nativePrec: 'bf16', variants: { fp8: 'RedHatAI/Mixtral-8x22B-Instruct-v0.1-FP8', int4: 'MaziyarPanahi/Mixtral-8x22B-Instruct-v0.1-AWQ' } },
  { id: 'mistral-small-3', hf: 'mistralai/Mistral-Small-3.2-24B-Instruct-2506', hfMirror: 'unsloth/Mistral-Small-3.2-24B-Instruct-2506',  family: 'Mistral',     name: 'Mistral Small 3.x 24B / Devstral',    params: 24,    layers: 40,  dModel: 5120,  nHeads: 32,  nKv: 8,  dHead: 128, maxCtx: 131072, nativePrec: 'bf16', variants: { fp8: 'stelterlab/Mistral-Small-3.2-24B-Instruct-2506-FP8', int4: 'Intel/Mistral-Small-3.2-24B-Instruct-2506-int4-AutoRound', fp4: 'RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4' } },
  { id: 'mistral-large-2', hf: 'mistralai/Mistral-Large-Instruct-2407', hfMirror: 'RedHatAI/Mistral-Large-Instruct-2407-FP8',  family: 'Mistral',     name: 'Mistral Large 2 123B',                params: 123,   layers: 88,  dModel: 12288, nHeads: 96,  nKv: 8,  dHead: 128, maxCtx: 131072, nativePrec: 'bf16', variants: { fp8: 'RedHatAI/Mistral-Large-Instruct-2407-FP8', int4: 'casperhansen/mistral-large-instruct-2407-awq' } },
  { id: 'qwen2.5-7b', hf: 'Qwen/Qwen2.5-7B-Instruct',       family: 'Qwen',        name: 'Qwen2.5 7B',                          params: 7.6,   layers: 28,  dModel: 3584,  nHeads: 28,  nKv: 4,  dHead: 128, maxCtx: 131072, nativeCtx: 32768, nativePrec: 'bf16', variants: { fp8: 'RedHatAI/Qwen2.5-7B-Instruct-FP8-dynamic', int8: 'Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8', int4: 'Qwen/Qwen2.5-7B-Instruct-AWQ' } },
  { id: 'qwen2.5-32b', hf: 'Qwen/Qwen2.5-Coder-32B-Instruct',      family: 'Qwen',        name: 'Qwen2.5 32B / Coder 32B',             params: 32.5,  layers: 64,  dModel: 5120,  nHeads: 40,  nKv: 8,  dHead: 128, maxCtx: 131072, nativeCtx: 32768, nativePrec: 'bf16', variants: { fp8: 'RedHatAI/Qwen2.5-Coder-32B-Instruct-FP8-dynamic', int8: 'Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int8', int4: 'Qwen/Qwen2.5-Coder-32B-Instruct-AWQ', fp4: 'drawais/Qwen2.5-Coder-32B-Instruct-NVFP4' } },
  { id: 'qwen2.5-72b', hf: 'Qwen/Qwen2.5-72B-Instruct',      family: 'Qwen',        name: 'Qwen2.5 72B',                         params: 72.7,  layers: 80,  dModel: 8192,  nHeads: 64,  nKv: 8,  dHead: 128, maxCtx: 131072, nativeCtx: 32768, nativePrec: 'bf16', variants: { fp8: 'RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic', int8: 'Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8', int4: 'Qwen/Qwen2.5-72B-Instruct-AWQ', fp4: 'enfuse/Qwen2.5-72B-Instruct-NVFP4' } },
  { id: 'qwen3-8b', hf: 'Qwen/Qwen3-8B',         family: 'Qwen',        name: 'Qwen3 8B',                            params: 8.2,   layers: 36,  dModel: 4096,  nHeads: 32,  nKv: 8,  dHead: 128, maxCtx: 131072, nativeCtx: 32768, nativeCtx: 40960, nativePrec: 'bf16', variants: { fp8: 'Qwen/Qwen3-8B-FP8', int4: 'Qwen/Qwen3-8B-AWQ', fp4: 'nvidia/Qwen3-8B-NVFP4' } },
  { id: 'qwen3-14b', hf: 'Qwen/Qwen3-14B',        family: 'Qwen',        name: 'Qwen3 14B',                           params: 14.8,  layers: 40,  dModel: 5120,  nHeads: 40,  nKv: 8,  dHead: 128, maxCtx: 131072, nativeCtx: 32768, nativeCtx: 40960, nativePrec: 'bf16', variants: { fp8: 'Qwen/Qwen3-14B-FP8', int4: 'Qwen/Qwen3-14B-AWQ', fp4: 'nvidia/Qwen3-14B-NVFP4' } },
  { id: 'qwen3-32b', hf: 'Qwen/Qwen3-32B',        family: 'Qwen',        name: 'Qwen3 32B',                           params: 32.8,  layers: 64,  dModel: 5120,  nHeads: 64,  nKv: 8,  dHead: 128, maxCtx: 131072, nativeCtx: 32768, nativeCtx: 40960, nativePrec: 'bf16', variants: { fp8: 'Qwen/Qwen3-32B-FP8', int4: 'Qwen/Qwen3-32B-AWQ', fp4: 'nvidia/Qwen3-32B-NVFP4' } },
  { id: 'qwen3-30b-a3b', hf: 'Qwen/Qwen3-30B-A3B-Instruct-2507',    family: 'Qwen',        name: 'Qwen3-30B-A3B (2507)',                params: 30.5,  active: 3.3, layers: 48, dModel: 2048, nHeads: 32, nKv: 4, dHead: 128, moe: { experts: 128, active: 8 }, maxCtx: 1010000, nativeCtx: 262144, nativePrec: 'bf16', variants: { fp8: 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8', int8: 'RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w8a8', int4: 'RedHatAI/Qwen3-30B-A3B-Instruct-2507-quantized.w4a16', fp4: 'nvidia/Qwen3-30B-A3B-NVFP4' } },
  { id: 'qwen3-235b-a22b', hf: 'Qwen/Qwen3-235B-A22B-Instruct-2507',  family: 'Qwen',        name: 'Qwen3-235B-A22B (2507)',              params: 235,   active: 22,  layers: 94, dModel: 4096, nHeads: 64, nKv: 4, dHead: 128, moe: { experts: 128, active: 8 }, maxCtx: 1010000, nativeCtx: 262144, nativePrec: 'bf16', variants: { fp8: 'Qwen/Qwen3-235B-A22B-Instruct-2507-FP8', int4: 'QuantTrio/Qwen3-235B-A22B-Instruct-2507-AWQ', fp4: 'nvidia/Qwen3-235B-A22B-NVFP4' } },
  { id: 'qwen3-coder-480b', hf: 'Qwen/Qwen3-Coder-480B-A35B-Instruct', family: 'Qwen',        name: 'Qwen3-Coder-480B-A35B',               params: 480,   active: 35,  layers: 62, dModel: 6144, nHeads: 96, nKv: 8, dHead: 128, moe: { experts: 160, active: 8 }, maxCtx: 1010000, nativeCtx: 262144, nativePrec: 'bf16', variants: { fp8: 'Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8', int4: 'QuantTrio/Qwen3-Coder-480B-A35B-Instruct-AWQ', fp4: 'nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4' } },
  { id: 'qwen3-next-80b', hf: 'Qwen/Qwen3-Next-80B-A3B-Instruct',   family: 'Qwen',        name: 'Qwen3-Next-80B-A3B / Qwen3-Coder-Next', params: 80,  active: 3,   layers: 48, dModel: 2048, nHeads: 16, nKv: 2, dHead: 256, attn: [{ n: 36, type: 'linear', nHeads: 32, dHead: 128 }, { n: 12, type: 'full' }], moe: { experts: 512, active: 10 }, statePerSeqMB: 40, maxCtx: 1010000, nativeCtx: 262144, note: 'Hybrid: 3 Gated DeltaNet layers per gated-attention layer', nativePrec: 'bf16', variants: { fp8: 'Qwen/Qwen3-Next-80B-A3B-Instruct-FP8', int8: 'RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w8a8', int4: 'RedHatAI/Qwen3-Next-80B-A3B-Instruct-quantized.w4a16', fp4: 'nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4' } },
  { id: 'deepseek-v3', hf: 'deepseek-ai/DeepSeek-V3.1',      family: 'DeepSeek',    name: 'DeepSeek-V3 / R1 / V3.1',             params: 671,   active: 37,  layers: 61, dModel: 7168, nHeads: 128, nKv: 128, dHead: 192, attn: [{ n: 61, type: 'mla', dc: 512, dr: 64 }], moe: { experts: 256, active: 8 }, maxCtx: 163840, note: 'Multi-head latent attention; config allows 160k positions, DeepSeek documents 128k', nativePrec: 'fp8', variants: { fp8: 'deepseek-ai/DeepSeek-V3.1', int4: 'QuantTrio/DeepSeek-V3.1-AWQ', fp4: 'nvidia/DeepSeek-V3.1-NVFP4' } },
  { id: 'deepseek-v3.2', hf: 'deepseek-ai/DeepSeek-V3.2', hfMirror: 'deepseek-ai/DeepSeek-V3.2-Exp',    family: 'DeepSeek',    name: 'DeepSeek-V3.2 (sparse attention)',    params: 671,   active: 37,  layers: 61, dModel: 7168, nHeads: 128, nKv: 128, dHead: 192, attn: [{ n: 61, type: 'mla', dc: 512, dr: 64 }], moe: { experts: 256, active: 8 }, sparse: { topk: 2048 }, maxCtx: 163840, note: 'MLA plus DeepSeek Sparse Attention (top-2048 tokens); config allows 160k positions, DeepSeek documents 128k', nativePrec: 'fp8', variants: { fp8: 'deepseek-ai/DeepSeek-V3.2', int4: 'QuantTrio/DeepSeek-V3.2-AWQ', fp4: 'nvidia/DeepSeek-V3.2-NVFP4' } },
  { id: 'kimi-k2', hf: 'moonshotai/Kimi-K2-Instruct-0905',          family: 'Moonshot',    name: 'Kimi K2 / K2 Thinking / K2.5',        params: 1026,  active: 32,  layers: 61, dModel: 7168, nHeads: 64, nKv: 64, dHead: 192, attn: [{ n: 61, type: 'mla', dc: 512, dr: 64 }], moe: { experts: 384, active: 8 }, maxCtx: 262144, note: 'Multi-head latent attention', nativePrec: 'fp8', variants: { fp8: 'moonshotai/Kimi-K2-Instruct-0905', int4: 'moonshotai/Kimi-K2-Thinking', fp4: 'nvidia/Kimi-K2-Thinking-NVFP4' } },
  { id: 'glm-4.5-air', hf: 'zai-org/GLM-4.5-Air',      family: 'Zhipu GLM',   name: 'GLM-4.5-Air 106B-A12B',               params: 106,   active: 12,  layers: 46, dModel: 4096, nHeads: 96, nKv: 8, dHead: 128, moe: { experts: 128, active: 8 }, maxCtx: 131072, nativePrec: 'bf16', variants: { fp8: 'zai-org/GLM-4.5-Air-FP8', int4: 'cyankiwi/GLM-4.5-Air-AWQ-4bit', fp4: 'Firworks/GLM-4.5-Air-nvfp4' } },
  { id: 'glm-4.6', hf: 'zai-org/GLM-4.6',          family: 'Zhipu GLM',   name: 'GLM-4.5 / 4.6 / 4.7 355B-A32B',       params: 355,   active: 32,  layers: 92, dModel: 5120, nHeads: 96, nKv: 8, dHead: 128, moe: { experts: 160, active: 8 }, maxCtx: 204800, maxCtx: 202752, nativePrec: 'bf16', variants: { fp8: 'zai-org/GLM-4.6-FP8', int4: 'QuantTrio/GLM-4.6-AWQ', fp4: 'RedHatAI/GLM-4.6-NVFP4' } },
  { id: 'gpt-oss-20b', hf: 'openai/gpt-oss-20b',      family: 'OpenAI',      name: 'gpt-oss-20b',                         params: 20.9,  active: 3.6, layers: 24, dModel: 2880, nHeads: 64, nKv: 8, dHead: 64, attn: [{ n: 12, type: 'swa', window: 128 }, { n: 12, type: 'full' }], moe: { experts: 32, active: 4 }, maxCtx: 131072, note: 'Alternating 128-token sliding window and full attention; MXFP4 experts', nativePrec: 'fp4 (MXFP4)', variants: { fp4: 'openai/gpt-oss-20b', int8: 'amd/gpt-oss-20b-BF16-w8a8-llmcompressor' } },
  { id: 'gpt-oss-120b', hf: 'openai/gpt-oss-120b',     family: 'OpenAI',      name: 'gpt-oss-120b',                        params: 116.8, active: 5.1, layers: 36, dModel: 2880, nHeads: 64, nKv: 8, dHead: 64, attn: [{ n: 18, type: 'swa', window: 128 }, { n: 18, type: 'full' }], moe: { experts: 128, active: 4 }, maxCtx: 131072, note: 'Alternating 128-token sliding window and full attention; MXFP4 experts', nativePrec: 'fp4 (MXFP4)', variants: { fp4: 'openai/gpt-oss-120b', bf16: 'axolotl-ai-co/gpt-oss-120b-dequantized' } },
  { id: 'gemma-3-4b', hf: 'google/gemma-3-4b-it', hfMirror: 'unsloth/gemma-3-4b-it',       family: 'Google Gemma', name: 'Gemma 3 4B',                         params: 4.3,   layers: 34,  dModel: 2560,  nHeads: 8,   nKv: 4,  dHead: 256, attn: [{ n: 29, type: 'swa', window: 1024 }, { n: 5, type: 'full' }], maxCtx: 131072, note: '5 local (1k window) layers per global layer', nativePrec: 'bf16', variants: { fp8: 'RedHatAI/gemma-3-4b-it-FP8-dynamic', int4: 'RedHatAI/gemma-3-4b-it-quantized.w4a16' } },
  { id: 'gemma-3-12b', hf: 'google/gemma-3-12b-it', hfMirror: 'unsloth/gemma-3-12b-it',      family: 'Google Gemma', name: 'Gemma 3 12B',                        params: 12.2,  layers: 48,  dModel: 3840,  nHeads: 16,  nKv: 8,  dHead: 256, attn: [{ n: 40, type: 'swa', window: 1024 }, { n: 8, type: 'full' }], maxCtx: 131072, note: '5 local (1k window) layers per global layer', nativePrec: 'bf16', variants: { fp8: 'RedHatAI/gemma-3-12b-it-FP8-dynamic', int4: 'RedHatAI/gemma-3-12b-it-quantized.w4a16' } },
  { id: 'gemma-3-27b', hf: 'google/gemma-3-27b-it', hfMirror: 'unsloth/gemma-3-27b-it',      family: 'Google Gemma', name: 'Gemma 3 27B',                        params: 27.4,  layers: 62,  dModel: 5376,  nHeads: 32,  nKv: 16, dHead: 128, attn: [{ n: 52, type: 'swa', window: 1024 }, { n: 10, type: 'full' }], maxCtx: 131072, note: '5 local (1k window) layers per global layer', nativePrec: 'bf16', variants: { fp8: 'RedHatAI/gemma-3-27b-it-FP8-dynamic', int8: 'RedHatAI/gemma-3-27b-it-quantized.w8a8', int4: 'RedHatAI/gemma-3-27b-it-quantized.w4a16', fp4: 'NeoChen1024/gemma-3-27b-it-NVFP4' } },
  { id: 'phi-4', hf: 'microsoft/phi-4',            family: 'Microsoft',   name: 'Phi-4 14B',                           params: 14.7,  layers: 40,  dModel: 5120,  nHeads: 40,  nKv: 10, dHead: 128, maxCtx: 16384, nativePrec: 'bf16', variants: { fp8: 'RedHatAI/phi-4-FP8-dynamic', int8: 'RedHatAI/phi-4-quantized.w8a8', int4: 'RedHatAI/phi-4-quantized.w4a16' } },
  { id: 'command-a', hf: 'CohereLabs/c4ai-command-a-03-2025', hfMirror: 'unsloth/c4ai-command-a-03-2025',        family: 'Cohere',      name: 'Command A 111B',                      params: 111,   layers: 64,  dModel: 12288, nHeads: 96,  nKv: 8,  dHead: 128, attn: [{ n: 48, type: 'swa', window: 4096 }, { n: 16, type: 'full' }], maxCtx: 262144, note: '3 sliding-window (4k) layers per full layer', nativePrec: 'bf16', variants: { fp8: 'aikitoria/c4ai-command-a-03-2025-FP8-Dynamic', int4: 'gghfez/c4ai-command-a-03-2025-AWQ', fp4: 'Firworks/c4ai-command-a-03-2025-nvfp4' } },
  { id: 'minimax-m1', hf: 'MiniMaxAI/MiniMax-M1-80k',       family: 'MiniMax',     name: 'MiniMax-M1 456B-A46B',                params: 456,   active: 45.9, layers: 80, dModel: 6144, nHeads: 64, nKv: 8, dHead: 128, attn: [{ n: 70, type: 'linear' }, { n: 10, type: 'full' }], moe: { experts: 32, active: 2 }, statePerSeqMB: 60, maxCtx: 10240000, note: 'Hybrid lightning attention (7:1); shape approximate', note: 'Hybrid lightning attention (7:1); config allows 10M positions, MiniMax documents 1M', nativePrec: 'bf16', variants: { int4: 'justinjja/MiniMax-M1-80k-W4A16-INT4' } },
];

if (typeof module !== 'undefined') module.exports = { LINKS, NETWORKS, ARCHS, ENGINE_FAMILY, ENGINES, HARDWARE, MODELS };
