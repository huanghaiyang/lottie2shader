import fs from 'node:fs';

const wasm = fs.readFileSync(process.env.L2S_WASM ?? 'dist/lottie2shader.wasm');
const { instance } = await WebAssembly.instantiate(wasm, {});
const ex = instance.exports;
const BATCH_WORDS = 64;

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
  const combined = (totalWords - 3 - batches * BATCH_WORDS - vertexCount * 2 - uvCount * 2) / 2;
  const kinds = {};
  for (let i = 0; i < batches; i++) {
    const base = 3 + i * BATCH_WORDS;
    const kind = words[base];
    kinds[kind] = (kinds[kind] || 0) + 1;
  }
  return { ok: true, batches, vertices: vertexCount, uvs: uvCount, residual: combined, kinds };
}

const dir = 'D:/sgames/lottie2shader/public/animations';
for (const name of fs.readdirSync(dir)) {
  const doc = JSON.parse(fs.readFileSync(`${dir}/${name}`, 'utf8'));
  const width = 512;
  const height = Math.round(512 * (doc.h / doc.w));
  const result = compile(doc, width, height);
  if (!result.ok) {
    console.log(name, 'COMPILE FAIL:', result.error);
    continue;
  }
  const frame = result.ip + (result.op - result.ip) * 0.25;
  const plan = render(frame);
  console.log(name, `fps=${result.fps} frames=${result.ip}-${result.op}`, JSON.stringify(plan));
}
