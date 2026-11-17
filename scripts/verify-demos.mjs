import fs from 'node:fs';

// 对指定的 Lottie JSON 做一次「编译 + 单帧渲染」冒烟，用于验证构造出来的测试数据
// 是否被 Rust 后端接受（真实样例之外的特性靠这类构造数据覆盖）。
// 用法：$env:L2S_WASM=<wasm>; $env:L2S_FILES="a.json;b.json"; node scripts/verify-demos.mjs

const wasm = fs.readFileSync(process.env.L2S_WASM ?? 'dist/lottie2shader.wasm');
const { instance } = await WebAssembly.instantiate(wasm, {});
const ex = instance.exports;
const BATCH_WORDS = 13 + 16 * 5 + 11;

function compile(json, width, height) {
  const bytes = Buffer.from(JSON.stringify(json));
  const ptr = ex.l2s_alloc(bytes.length);
  new Uint8Array(ex.memory.buffer, ptr, bytes.length).set(bytes);
  const ok = ex.l2s_compile(ptr, bytes.length, width, height);
  if (!ok) {
    const len = ex.l2s_error_len();
    const ep = ex.l2s_error_ptr();
    const error = Buffer.from(new Uint8Array(ex.memory.buffer, ep, len)).toString();
    ex.l2s_free(ptr, bytes.length);
    return { ok: false, error };
  }
  ex.l2s_free(ptr, bytes.length);
  return { ok: true, fps: ex.l2s_fps(), ip: ex.l2s_ip(), op: ex.l2s_op() };
}

function render(frame) {
  const ok = ex.l2s_render(frame);
  if (!ok) return { ok: false };
  const totalWords = ex.l2s_result_len() / 4;
  const ptr = ex.l2s_result_ptr();
  const words = new Uint32Array(ex.memory.buffer, ptr, totalWords);
  const batches = words[0];
  const vertexCount = words[1];
  const uvCount = words[2];
  const tail = 3 + batches * BATCH_WORDS + vertexCount * 2 + uvCount * 2;
  const passCount = words[tail] ?? 0;
  const maskStart = tail + 1 + passCount * 10;
  const maskCount = words[maskStart] ?? 0;
  const passes = [];
  for (let i = 0; i < passCount; i++) {
    const base = tail + 1 + i * 10;
    passes.push({
      id: words[base],
      sigmaX: new Uint32Array([words[base + 4]]).map((w) => w)[0],
      wrap: words[base + 3],
    });
  }
  return {
    ok: true,
    batches,
    vertices: vertexCount,
    passes: passCount,
    passSigmaX: passes.map((p) => {
      const buf = new ArrayBuffer(4);
      new Uint32Array(buf)[0] = p.sigmaX;
      return Math.round(new Float32Array(buf)[0] * 1000) / 1000;
    }),
    masks: maskCount,
    residual: totalWords - (maskStart + 1 + maskCount * 6),
  };
}

const files = (process.env.L2S_FILES ?? '').split(';').filter(Boolean);
for (const path of files) {
  const doc = JSON.parse(fs.readFileSync(path, 'utf8'));
  const width = 512;
  const height = Math.round(512 * (doc.h / doc.w));
  const result = compile(doc, width, height);
  if (!result.ok) {
    console.log(path, 'COMPILE FAIL:', result.error);
    continue;
  }
  const frame = result.ip + (result.op - result.ip) * 0.25;
  console.log(path, `fps=${result.fps} frames=${result.ip}-${result.op}`, JSON.stringify(render(frame)));
}
