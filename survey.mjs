// 统计真实样例里各类不支持特性的出现情况（只扫顶层图层与一层 assets，避免引用环）。
import fs from 'node:fs';

const dir = 'D:/sgames/lottie2shader/public/animations';
for (const f of fs.readdirSync(dir)) {
  const d = JSON.parse(fs.readFileSync(`${dir}/${f}`, 'utf8'));
  const layerTy = {};
  const efTy = {};
  let efLayers = 0;
  let maxEf = 0;
  let maskX = 0;
  let expr = 0;
  const visit = (layers) => {
    for (const l of layers ?? []) {
      layerTy[l.ty] = (layerTy[l.ty] ?? 0) + 1;
      const ef = (l.ef ?? []).filter((e) => e.en !== 0 && e.en !== false);
      if (ef.length) {
        efLayers += 1;
        maxEf = Math.max(maxEf, ef.length);
        for (const e of ef) efTy[e.ty] = (efTy[e.ty] ?? 0) + 1;
      }
      for (const m of l.masksProperties ?? []) if (m.x && (m.x.k !== undefined || m.x.a !== undefined)) maskX += 1;
    }
  };
  visit(d.layers);
  for (const a of d.assets ?? []) visit(a.layers);
  const scan = (o) => {
    if (Array.isArray(o)) {
      for (const item of o) scan(item);
    } else if (o && typeof o === 'object') {
      for (const k of Object.keys(o)) {
        if (k === 'x' && typeof o[k] === 'string') expr += 1;
        scan(o[k]);
      }
    }
  };
  scan(d.layers);
  console.log(
    f,
    '图层类型',
    JSON.stringify(layerTy),
    '带特效图层',
    efLayers,
    '单层特效数上限',
    maxEf,
    '特效类型',
    JSON.stringify(efTy),
    '蒙版x',
    maskX,
    '表达式',
    expr,
  );
}
