/* arcade/scenes.js - the built-in scenes: saved setups to step through with a clicker (PageDown / PageUp) or keys 1-9.
 * Each scene is applied on top of the planner defaults (common.js DEFAULTS), so it always looks the same.
 * `baseline`: 'pin' pins the scene as the baseline every later number is compared with, 'clear' removes the baseline,
 * 'keep' (default) leaves it. Edit freely for your own talk; Shift+1-9 on stage saves over a slot in this browser.
 */
const SCENES = [
  {
    name: 'It works on my laptop', baseline: 'clear',
    state: { engine: 'ollama', model: 'llama-3.1-8b', wPrec: 'int4', kvPrec: 'bf16', hw: 'rtx-4090', count: 1, nodeGpus: 8, users: 1, activityPreset: 'chat', activity: 10, ctx: 8192, prefix: 1024, newPrompt: 300, output: 400, retention: 'host', target: 20 },
  },
  {
    name: 'Share it with 50 colleagues', baseline: 'pin',
    state: { engine: 'ollama', model: 'llama-3.1-8b', wPrec: 'int4', kvPrec: 'bf16', hw: 'rtx-4090', count: 1, nodeGpus: 8, users: 50, activityPreset: 'chat', activity: 10, ctx: 8192, prefix: 1024, newPrompt: 300, output: 400, retention: 'host', target: 20 },
  },
  {
    name: 'Same GPU, a serving engine',
    state: { engine: 'vllm', model: 'llama-3.1-8b', wPrec: 'int4', kvPrec: 'bf16', hw: 'rtx-4090', count: 1, nodeGpus: 8, users: 50, activityPreset: 'chat', activity: 10, ctx: 8192, prefix: 1024, newPrompt: 300, output: 400, retention: 'host', target: 20 },
  },
  {
    name: 'A bigger brain: 70B', baseline: 'clear',
    state: { engine: 'vllm', model: 'llama-3.3-70b', wPrec: 'bf16', kvPrec: 'bf16', hw: 'h100-sxm', count: 1, users: 50, activityPreset: 'chat', activity: 10, ctx: 8192, prefix: 1024, newPrompt: 300, output: 400, retention: 'host', target: 20 },
  },
  {
    name: 'FP8 on two GPUs', baseline: 'pin',
    state: { engine: 'vllm', model: 'llama-3.3-70b', wPrec: 'fp8', kvPrec: 'bf16', hw: 'h100-sxm', count: 2, users: 50, activityPreset: 'chat', activity: 10, ctx: 8192, prefix: 1024, newPrompt: 300, output: 400, retention: 'host', target: 20 },
  },
  {
    name: 'Long documents: 128k context',
    state: { engine: 'vllm', model: 'llama-3.3-70b', wPrec: 'fp8', kvPrec: 'bf16', hw: 'h100-sxm', count: 2, users: 50, activityPreset: 'chat', activity: 10, ctx: 131072, prefix: 1024, newPrompt: 2000, output: 500, retention: 'host', target: 20, prefixCache: false },
  },
  {
    name: 'FP8 KV cache and prefix caching',
    state: { engine: 'vllm', model: 'llama-3.3-70b', wPrec: 'fp8', kvPrec: 'fp8', hw: 'h100-sxm', count: 2, users: 50, activityPreset: 'chat', activity: 10, ctx: 131072, prefix: 1024, newPrompt: 2000, output: 500, retention: 'host', target: 20, prefixCache: true },
  },
  {
    name: 'Coding agents for 500 people', baseline: 'pin', autoBuild: true,
    state: { engine: 'vllm', model: 'llama-3.3-70b', wPrec: 'fp8', kvPrec: 'fp8', hw: 'h100-sxm', count: 8, users: 500, activityPreset: 'agents', activity: 60, ctx: 131072, prefix: 4096, newPrompt: 4000, output: 1000, retention: 'host', target: 30, prefixCache: true },
  },
  {
    name: 'A frontier MoE on Blackwell', autoBuild: true,
    state: { engine: 'vllm', model: 'deepseek-v3', wPrec: 'fp4', kvPrec: 'fp8', hw: 'b200', count: 8, dpAttn: true, users: 500, activityPreset: 'agents', activity: 60, ctx: 131072, prefix: 4096, newPrompt: 4000, output: 1000, retention: 'host', target: 30, prefixCache: true },
  },
];
if (typeof module !== 'undefined') module.exports = SCENES;
