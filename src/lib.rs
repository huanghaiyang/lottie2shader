//! lottie2shader —— 把 Lottie 形状动画编译成 GPU 三角面片渲染计划。
//!
//! 设计目标：
//! - 不依赖 wasm-bindgen：以裸 WASM 导出函数 + 线性内存交换数据，前端直接
//!   `WebAssembly.instantiate` 即可加载；
//! - 解析 Lottie JSON，在 Rust 侧完成路径求值、贝塞尔曲线离散化与三角化，
//!   几何一次性烘焙到图层局部坐标；
//! - 每帧只做「矩阵变换 + 顶点输出」，由 WebGL2 着色器完成光栅化、渐变与合成，
//!   从而把 Lottie 的绘制负担从 Canvas2D 移到 GPU；
//! - 以二进制批次格式输出：批次头（画笔类型/颜色/渐变/顶点区间）+ 顶点坐标，
//!   前端按批次 drawArrays，避免逐帧解析 JSON。
//!
//! 当前支持范围（形状动画 MVP）：
//! - 形状图层（ty=4）及其嵌套组（gr / tr）；
//! - 矩形 rc、椭圆 el、贝塞尔路径 sh；
//! - 纯色填充 fl、纯色描边 st、线性/径向渐变填充 gf；
//! - 图层与组的变换、不透明度关键帧（位置/锚点/缩放/旋转）；
//! - 图层父子关系（parent）；
//! - 路径形变关键帧暂取首帧、文本/图片/遮罩/表达式暂不支持（由前端自动回退）。

use lyon::math::Point;
use lyon::path::Path;
use lyon::tessellation::geometry_builder::{BuffersBuilder, VertexBuffers};
use lyon::tessellation::{
  FillOptions, FillRule, FillTessellator, FillVertex, LineCap, LineJoin, StrokeOptions,
  StrokeTessellator, StrokeVertex,
};
use serde_json::Value;

/// 单个批次最多承载的渐变色标数量，与前端着色器 uniform 数组长度保持一致
const MAX_STOPS: usize = 8;
/// 批次头固定占用字（u32）数：13 个字段 + 8 个色标 × 5 个分量
const BATCH_WORDS: usize = 13 + MAX_STOPS * 5;
/// 贝塞尔离散化的世界坐标容差（远小于 1 像素，保证矢量边缘足够平滑）
const TOLERANCE: f32 = 0.15;

// ---------------------------------------------------------------------------
// 2D 仿射矩阵
// ---------------------------------------------------------------------------

/// 2D 仿射矩阵，按 canvas 约定排列：[a, b, c, d, e, f]
/// 变换公式：x' = a*x + c*y + e，y' = b*x + d*y + f
#[derive(Clone, Copy)]
struct Mat([f32; 6]);

impl Mat {
  fn identity() -> Self {
    Mat([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
  }

  fn translate(x: f32, y: f32) -> Self {
    Mat([1.0, 0.0, 0.0, 1.0, x, y])
  }

  fn scale(x: f32, y: f32) -> Self {
    Mat([x, 0.0, 0.0, y, 0.0, 0.0])
  }

  fn rotate(rad: f32) -> Self {
    let (s, c) = rad.sin_cos();
    Mat([c, s, -s, c, 0.0, 0.0])
  }

  /// 矩阵乘法：标准复合 `self ∘ other`（先应用 other，再应用 self）
  fn mul(&self, other: &Mat) -> Mat {
    let a = self.0;
    let b = other.0;
    Mat([
      a[0] * b[0] + a[2] * b[1],
      a[1] * b[0] + a[3] * b[1],
      a[0] * b[2] + a[2] * b[3],
      a[1] * b[2] + a[3] * b[3],
      a[0] * b[4] + a[2] * b[5] + a[4],
      a[1] * b[4] + a[3] * b[5] + a[5],
    ])
  }

  fn apply(&self, p: [f32; 2]) -> [f32; 2] {
    [
      self.0[0] * p[0] + self.0[2] * p[1] + self.0[4],
      self.0[1] * p[0] + self.0[3] * p[1] + self.0[5],
    ]
  }

  /// 矩阵的平均缩放因子，用于描边宽度随变换缩放
  fn scale_factor(&self) -> f32 {
    (self.0[0] * self.0[3] - self.0[1] * self.0[2]).abs().sqrt().max(0.0001)
  }
}

// ---------------------------------------------------------------------------
// 属性求值（静态值 / 关键帧）
// ---------------------------------------------------------------------------

fn as_f32(v: Option<&Value>, def: f32) -> f32 {
  v.and_then(|x| x.as_f64()).map(|x| x as f32).unwrap_or(def)
}

fn is_animated(prop: Option<&Value>) -> bool {
  // Lottie 的动画标记可能是布尔 `true`，也可能是数字 `1`（不同导出器差异），两者都要认
  match prop.and_then(|p| p.get("a")) {
    Some(Value::Bool(flag)) => *flag,
    Some(Value::Number(num)) => num.as_f64().unwrap_or(0.0) > 0.0,
    _ => false,
  }
}

/// 关键帧缓动的 x 轴求解：给定时间进度 t，反解三次贝塞尔曲线的 x 参数
fn bezier_x(t: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
  let _ = (y1, y2);
  let mut lo = 0.0f32;
  let mut hi = 1.0f32;
  let mut u = t;
  for _ in 0..12 {
    // 三次贝塞尔单参数求值
    let inv = 1.0 - u;
    let x = 3.0 * inv * inv * u * x1 + 3.0 * inv * u * u * x2 + u * u * u;
    if (x - t).abs() < 0.0005 {
      break;
    }
    if x < t {
      lo = u;
    } else {
      hi = u;
    }
    u = (lo + hi) * 0.5;
  }
  u
}

fn bezier_y(t: f32, x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
  let u = bezier_x(t, x1, y1, x2, y2);
  let inv = 1.0 - u;
  3.0 * inv * inv * u * y1 + 3.0 * inv * u * u * y2 + u * u * u
}

/// 从 `{x:[..], y:[..]}` 中读取缓动控制点；部分导出把百分比写成 0~100，这里统一归一化
fn easing_pair(v: Option<&Value>) -> (f32, f32) {
  let x = v
    .and_then(|e| e.get("x"))
    .and_then(|x| x.as_array())
    .and_then(|x| x.first())
    .and_then(|x| x.as_f64())
    .unwrap_or(0.0) as f32;
  let y = v
    .and_then(|e| e.get("y"))
    .and_then(|y| y.as_array())
    .and_then(|y| y.first())
    .and_then(|y| y.as_f64())
    .unwrap_or(0.0) as f32;
  let norm = if x > 1.0 || y > 1.0 { 100.0 } else { 1.0 };
  (x / norm, y / norm)
}

fn value_vec(v: Option<&Value>, dim: usize, def: &[f32]) -> Vec<f32> {
  match v {
    Some(Value::Number(n)) => vec![n.as_f64().unwrap_or(0.0) as f32; dim],
    Some(Value::Array(arr)) => (0..dim)
      .map(|i| {
        arr
          .get(i)
          .and_then(|x| x.as_f64())
          .map(|x| x as f32)
          .unwrap_or_else(|| def.get(i).copied().unwrap_or(0.0))
      })
      .collect(),
    _ => def.to_vec(),
  }
}

/// 求值一个 Lottie 属性；支持静态数值、静态数组与关键帧插值
fn prop_vec(prop: Option<&Value>, frame: f32, dim: usize, def: &[f32]) -> Vec<f32> {
  let Some(p) = prop else {
    return def.to_vec();
  };
  let k = p.get("k");
  if !is_animated(Some(p)) {
    return value_vec(k, dim, def);
  }
  let Some(kfs) = k.and_then(|k| k.as_array()) else {
    return value_vec(k, dim, def);
  };
  if kfs.is_empty() {
    return def.to_vec();
  }
  let time = frame;
  let first = &kfs[0];
  if time <= as_f32(first.get("t"), 0.0) {
    return value_vec(first.get("s"), dim, def);
  }
  for i in 0..kfs.len() - 1 {
    let a = &kfs[i];
    let b = &kfs[i + 1];
    let ta = as_f32(a.get("t"), 0.0);
    let tb = as_f32(b.get("t"), ta + 1.0);
    if time >= ta && time < tb {
      if a.get("h").and_then(|h| h.as_bool()).unwrap_or(false) {
        return value_vec(a.get("s"), dim, def);
      }
      let raw = if tb > ta { (time - ta) / (tb - ta) } else { 0.0 };
      let (x1, y1) = easing_pair(a.get("o"));
      let (x2, y2) = easing_pair(b.get("i"));
      let eased = if x1 == 0.0 && y1 == 0.0 && x2 == 1.0 && y2 == 1.0 {
        raw
      } else {
        bezier_y(raw, x1, y1, x2, y2)
      };
      let start = value_vec(a.get("s"), dim, def);
      let end = value_vec(
        a.get("e").or_else(|| b.get("s")),
        dim,
        &start,
      );
      return (0..dim)
        .map(|j| start.get(j).copied().unwrap_or(0.0)
          + (end.get(j).copied().unwrap_or(0.0) - start.get(j).copied().unwrap_or(0.0)) * eased)
        .collect();
    }
  }
  value_vec(kfs.last().and_then(|kf| kf.get("s").or_else(|| kf.get("e"))), dim, def)
}

fn prop_f32(prop: Option<&Value>, frame: f32, def: f32) -> f32 {
  prop_vec(prop, frame, 1, &[def])
    .first()
    .copied()
    .unwrap_or(def)
}

/// 把图层 / 组变换解算成矩阵（锚点 → 缩放 → 旋转 → 位移）
fn tr_matrix(tr: &Value, frame: f32) -> Mat {
  let p = prop_vec(tr.get("p"), frame, 2, &[0.0, 0.0]);
  let a = prop_vec(tr.get("a"), frame, 2, &[0.0, 0.0]);
  let s = prop_vec(tr.get("s"), frame, 2, &[100.0, 100.0]);
  let r = prop_f32(tr.get("r"), frame, 0.0).to_radians();
  Mat::translate(p[0], p[1])
    .mul(&Mat::rotate(r))
    .mul(&Mat::scale(s[0] / 100.0, s[1] / 100.0))
    .mul(&Mat::translate(-a[0], -a[1]))
}

// ---------------------------------------------------------------------------
// 形状数据模型
// ---------------------------------------------------------------------------

/// 一种画笔（纯色 / 线性渐变 / 径向渐变）
struct Paint {
  kind: u32,
  color: [f32; 4],
  alpha: f32,
  p0: [f32; 2],
  p1: [f32; 2],
  stops: Vec<[f32; 5]>,
}

/// 一段已烘焙到局部坐标的三角形网格
struct Geom {
  verts: Vec<[f32; 2]>,
  paint: usize,
}

struct Layer {
  geoms: Vec<Geom>,
  paints: Vec<Paint>,
  ks: Value,
  parent: Option<usize>,
  ip: f32,
  op: f32,
}

struct Compiled {
  doc_w: f32,
  doc_h: f32,
  fr: f32,
  ip: f32,
  op: f32,
  width: f32,
  height: f32,
  layers: Vec<Layer>,
}

// ---------------------------------------------------------------------------
// 路径构建
// ---------------------------------------------------------------------------

fn to_point(mat: &Mat, p: [f32; 2]) -> Point {
  let t = mat.apply(p);
  Point::new(t[0], t[1])
}

/// 由顶点 + 入/出切线构建三次贝塞尔路径（切线为相对顶点的偏移量）
fn bezier_path(
  verts: &[[f32; 2]],
  ins: &[[f32; 2]],
  outs: &[[f32; 2]],
  closed: bool,
  mat: &Mat,
) -> Option<Path> {
  if verts.len() < 2 {
    return None;
  }
  let mut builder = Path::builder();
  builder.begin(to_point(mat, verts[0]));
  let n = verts.len();
  let segments = if closed { n } else { n - 1 };
  for i in 0..segments {
    let j = (i + 1) % n;
    let c1 = [
      verts[i][0] + outs.get(i).copied().unwrap_or([0.0, 0.0])[0],
      verts[i][1] + outs.get(i).copied().unwrap_or([0.0, 0.0])[1],
    ];
    let c2 = [
      verts[j][0] + ins.get(j).copied().unwrap_or([0.0, 0.0])[0],
      verts[j][1] + ins.get(j).copied().unwrap_or([0.0, 0.0])[1],
    ];
    let straight = (c1[0] - verts[i][0]).abs() < 0.01
      && (c1[1] - verts[i][1]).abs() < 0.01
      && (c2[0] - verts[j][0]).abs() < 0.01
      && (c2[1] - verts[j][1]).abs() < 0.01;
    if straight {
      builder.line_to(to_point(mat, verts[j]));
    } else {
      builder.cubic_bezier_to(to_point(mat, c1), to_point(mat, c2), to_point(mat, verts[j]));
    }
  }
  builder.end(closed);
  Some(builder.build())
}

fn arr2(v: Option<&Value>) -> Vec<[f32; 2]> {
  v.and_then(|x| x.as_array())
    .map(|arr| {
      arr
        .iter()
        .map(|p| {
          let a = p.as_array();
          [
            a.and_then(|a| a.first()).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
            a.and_then(|a| a.get(1)).and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
          ]
        })
        .collect()
    })
    .unwrap_or_default()
}

/// 取属性的首个关键帧值（用于编译期烘焙路径/尺寸）
fn first_keyframe_value(prop: &Value) -> Option<Value> {
  let k = prop.get("k")?;
  if is_animated(Some(prop)) {
    let first = k.as_array()?.first()?;
    first.get("s").or_else(|| first.get("e")).cloned()
  } else {
    Some(k.clone())
  }
}

fn path_from_shape(item: &Value, mat: &Mat) -> Option<Path> {
  let k = first_keyframe_value(item.get("ks")?)?;
  // 关键帧路径的 `s` 是「路径对象数组」（如 `[{c,i,o,v}]`），静态路径则直接是对象；
  // 这里统一解包到单个路径对象，否则动画路径会被误判为空。
  let path_data = if k.is_array() {
    k.as_array()?.first()?.clone()
  } else {
    k
  };
  let verts = arr2(path_data.get("v"));
  let ins = arr2(path_data.get("i"));
  let outs = arr2(path_data.get("o"));
  let closed = path_data.get("c").and_then(|c| c.as_bool()).unwrap_or(false);
  bezier_path(&verts, &ins, &outs, closed, mat)
}

fn path_from_ellipse(item: &Value, mat: &Mat) -> Option<Path> {
  let p = prop_vec(item.get("p"), 0.0, 2, &[0.0, 0.0]);
  let s = prop_vec(item.get("s"), 0.0, 2, &[0.0, 0.0]);
  let rx = (s[0].abs() / 2.0).max(0.01);
  let ry = (s[1].abs() / 2.0).max(0.01);
  let k = 0.552_284_8;
  let cx = p[0];
  let cy = p[1];
  let verts = [
    [cx + rx, cy],
    [cx, cy + ry],
    [cx - rx, cy],
    [cx, cy - ry],
  ];
  // 控制点相对顶点的偏移：右→下→左→上顺时针（Y 轴向下）。
  // outs 是本段起点处的切向，ins 是下一顶点处的入切向，必须指向曲线前进方向。
  let ins = [
    [0.0, -ry * k],
    [rx * k, 0.0],
    [0.0, ry * k],
    [-rx * k, 0.0],
  ];
  let outs = [
    [0.0, ry * k],
    [-rx * k, 0.0],
    [0.0, -ry * k],
    [rx * k, 0.0],
  ];
  bezier_path(&verts, &ins, &outs, true, mat)
}

fn path_from_rect(item: &Value, mat: &Mat) -> Option<Path> {
  let p = prop_vec(item.get("p"), 0.0, 2, &[0.0, 0.0]);
  let s = prop_vec(item.get("s"), 0.0, 2, &[0.0, 0.0]);
  let r = prop_f32(item.get("r"), 0.0, 0.0).max(0.0);
  let hw = s[0].abs() / 2.0;
  let hh = s[1].abs() / 2.0;
  let rr = r.min(hw).min(hh);
  let (cx, cy) = (p[0], p[1]);
  if rr <= 0.01 {
    let verts = [
      [cx - hw, cy - hh],
      [cx + hw, cy - hh],
      [cx + hw, cy + hh],
      [cx - hw, cy + hh],
    ];
    return bezier_path(&verts, &[], &[], true, mat);
  }
  let k = 0.552_284_8 * rr;
  // 圆角矩形：8 个顶点 + 四段直线 + 四段四分之一圆弧
  let verts = [
    [cx - hw + rr, cy - hh],
    [cx + hw - rr, cy - hh],
    [cx + hw, cy - hh + rr],
    [cx + hw, cy + hh - rr],
    [cx + hw - rr, cy + hh],
    [cx - hw + rr, cy + hh],
    [cx - hw, cy + hh - rr],
    [cx - hw, cy - hh + rr],
  ];
  let outs = [
    [0.0, 0.0],
    [k, 0.0],
    [0.0, 0.0],
    [0.0, k],
    [0.0, 0.0],
    [-k, 0.0],
    [0.0, 0.0],
    [0.0, -k],
  ];
  let ins = [
    [0.0, 0.0],
    [-k, 0.0],
    [0.0, 0.0],
    [0.0, -k],
    [0.0, 0.0],
    [k, 0.0],
    [0.0, 0.0],
    [0.0, k],
  ];
  bezier_path(&verts, &ins, &outs, true, mat)
}

// ---------------------------------------------------------------------------
// 三角化
// ---------------------------------------------------------------------------

fn tessellate_fill(paths: &[Path], rule: FillRule, paint: usize, geoms: &mut Vec<Geom>) {
  let mut verts: Vec<[f32; 2]> = Vec::new();
  for path in paths {
    let mut geometry: VertexBuffers<Point, u32> = VertexBuffers::new();
    let mut tessellator = FillTessellator::new();
    let options = FillOptions::tolerance(TOLERANCE).with_fill_rule(rule);
    let ok = tessellator
      .tessellate_path(
        path,
        &options,
        &mut BuffersBuilder::new(&mut geometry, |v: FillVertex| v.position()),
      )
      .is_ok();
    if !ok {
      continue;
    }
    for tri in geometry.indices.chunks(3) {
      if tri.len() < 3 {
        continue;
      }
      for &idx in tri {
        if let Some(p) = geometry.vertices.get(idx as usize) {
          verts.push([p.x, p.y]);
        }
      }
    }
  }
  if !verts.is_empty() {
    geoms.push(Geom { verts, paint });
  }
}

fn tessellate_stroke(
  paths: &[Path],
  width: f32,
  join: LineJoin,
  cap: LineCap,
  paint: usize,
  geoms: &mut Vec<Geom>,
) {
  let mut verts: Vec<[f32; 2]> = Vec::new();
  for path in paths {
    let mut geometry: VertexBuffers<Point, u32> = VertexBuffers::new();
    let mut tessellator = StrokeTessellator::new();
    let options = StrokeOptions::tolerance(TOLERANCE)
      .with_line_width(width.max(0.1))
      .with_line_join(join)
      .with_line_cap(cap);
    let ok = tessellator
      .tessellate_path(
        path,
        &options,
        &mut BuffersBuilder::new(&mut geometry, |v: StrokeVertex| v.position()),
      )
      .is_ok();
    if !ok {
      continue;
    }
    for tri in geometry.indices.chunks(3) {
      if tri.len() < 3 {
        continue;
      }
      for &idx in tri {
        if let Some(p) = geometry.vertices.get(idx as usize) {
          verts.push([p.x, p.y]);
        }
      }
    }
  }
  if !verts.is_empty() {
    geoms.push(Geom { verts, paint });
  }
}

// ---------------------------------------------------------------------------
// 图层解析
// ---------------------------------------------------------------------------

fn solid_paint(item: &Value, alpha: f32, frame: f32) -> Paint {
  let color = prop_vec(item.get("c"), frame, 4, &[0.0, 0.0, 0.0, 1.0]);
  Paint {
    kind: 0,
    color: [color[0], color[1], color[2], color.get(3).copied().unwrap_or(1.0)],
    alpha: alpha * prop_f32(item.get("o"), frame, 100.0) / 100.0,
    p0: [0.0, 0.0],
    p1: [1.0, 1.0],
    stops: Vec::new(),
  }
}

fn gradient_paint(item: &Value, alpha: f32, frame: f32) -> Paint {
  let s = prop_vec(item.get("s"), frame, 2, &[0.0, 0.0]);
  let e = prop_vec(item.get("e"), frame, 2, &[1.0, 0.0]);
  // Lottie：t=1 线性，t=2 径向；部分导出用 r=1 表示径向
  let is_radial = as_f32(item.get("t"), 0.0) == 2.0
    || as_f32(item.get("r"), 0.0) >= 1.0;
  let mut stops: Vec<[f32; 5]> = Vec::new();
  if let Some(g) = item.get("g") {
    let count = as_f32(g.get("p"), 0.0) as usize;
    let k = g.get("k");
    let raw = if is_animated(k) {
      k.and_then(|k| k.as_array())
        .and_then(|arr| arr.first())
        .and_then(|kf| kf.get("s").or_else(|| kf.get("e")))
        .cloned()
    } else {
      k.and_then(|k| k.get("k")).cloned()
    };
    let values: Vec<f32> = raw
      .and_then(|v| v.as_array().cloned())
      .map(|arr| {
        arr
          .iter()
          .map(|x| x.as_f64().unwrap_or(0.0) as f32)
          .collect()
      })
      .unwrap_or_default();
    // 色标可能是 4 分量（rgb）或 5 分量（rgba），按总长度判断
    let stride = if count > 0 && values.len() == count * 5 { 5 } else { 4 };
    let mut i = 0;
    while i + stride <= values.len() && stops.len() < MAX_STOPS {
      let offset = values[i];
      let r = values[i + 1];
      let g = values[i + 2];
      let b = values[i + 3];
      let a = if stride == 5 { values[i + 4] } else { 1.0 };
      stops.push([offset, r, g, b, a]);
      i += stride;
    }
  }
  if stops.is_empty() {
    stops.push([0.0, 0.0, 0.0, 0.0, 1.0]);
  }
  Paint {
    kind: if is_radial { 2 } else { 1 },
    color: [0.0, 0.0, 0.0, 1.0],
    alpha: alpha * prop_f32(item.get("o"), frame, 100.0) / 100.0,
    p0: [s[0], s[1]],
    p1: [e[0], e[1]],
    stops,
  }
}

fn walk_items(
  items: &[Value],
  base: Mat,
  alpha: f32,
  paints: &mut Vec<Paint>,
  geoms: &mut Vec<Geom>,
  frame: f32,
) {
  // 组内 `tr` 决定本组变换与不透明度
  let mut mat = base;
  let mut group_alpha = alpha;
  if let Some(tr) = items.iter().find(|i| i.get("ty").and_then(|t| t.as_str()) == Some("tr")) {
    mat = base.mul(&tr_matrix(tr, frame));
    group_alpha *= prop_f32(tr.get("o"), frame, 100.0) / 100.0;
  }

  // 组内已收集的路径，供后续填充 / 描边画笔使用
  let mut paths: Vec<Path> = Vec::new();

  for item in items {
    let ty = item.get("ty").and_then(|t| t.as_str()).unwrap_or("");
    match ty {
      "gr" => {
        if let Some(child) = item.get("it").and_then(|i| i.as_array()) {
          walk_items(child, mat, group_alpha, paints, geoms, frame);
        }
      }
      "sh" => {
        if let Some(p) = path_from_shape(item, &mat) {
          paths.push(p);
        }
      }
      "el" => {
        if let Some(p) = path_from_ellipse(item, &mat) {
          paths.push(p);
        }
      }
      "rc" => {
        if let Some(p) = path_from_rect(item, &mat) {
          paths.push(p);
        }
      }
      "fl" => {
        let paint = solid_paint(item, group_alpha, frame);
        let index = paints.len();
        paints.push(paint);
        let rule = if as_f32(item.get("r"), 1.0) >= 2.0 {
          FillRule::EvenOdd
        } else {
          FillRule::NonZero
        };
        tessellate_fill(&paths, rule, index, geoms);
      }
      "st" => {
        let paint = solid_paint(item, group_alpha, frame);
        let index = paints.len();
        paints.push(paint);
        let width = prop_f32(item.get("w"), frame, 1.0) * mat.scale_factor();
        let join = match as_f32(item.get("lj"), 2.0) as i32 {
          1 => LineJoin::Miter,
          3 => LineJoin::Bevel,
          _ => LineJoin::Round,
        };
        let cap = match as_f32(item.get("lc"), 2.0) as i32 {
          1 => LineCap::Butt,
          3 => LineCap::Square,
          _ => LineCap::Round,
        };
        tessellate_stroke(&paths, width, join, cap, index, geoms);
      }
      "gf" => {
        let paint = gradient_paint(item, group_alpha, frame);
        let index = paints.len();
        paints.push(paint);
        let rule = if as_f32(item.get("r"), 1.0) >= 2.0 {
          FillRule::EvenOdd
        } else {
          FillRule::NonZero
        };
        tessellate_fill(&paths, rule, index, geoms);
      }
      "gs" => {
        // 渐变描边暂用首色标近似为纯色描边
        let gradient = gradient_paint(item, group_alpha, frame);
        let color = gradient
          .stops
          .first()
          .map(|s| [s[1], s[2], s[3], s[4]])
          .unwrap_or([0.0, 0.0, 0.0, 1.0]);
        let index = paints.len();
        paints.push(Paint {
          kind: 0,
          color,
          alpha: gradient.alpha,
          p0: gradient.p0,
          p1: gradient.p1,
          stops: Vec::new(),
        });
        let width = prop_f32(item.get("w"), frame, 1.0) * mat.scale_factor();
        tessellate_stroke(&paths, width, LineJoin::Round, LineCap::Round, index, geoms);
      }
      _ => {}
    }
  }
}

fn compile_layer(layer: &Value, frame: f32) -> Option<Layer> {
  if layer.get("ty").and_then(|t| t.as_f64()).unwrap_or(0.0) != 4.0 {
    return None;
  }
  if layer.get("hd").and_then(|h| h.as_bool()).unwrap_or(false) {
    return None;
  }
  // 轨道遮罩的源图层本身不参与绘制（遮罩合成暂不支持）
  if layer.get("td").and_then(|t| t.as_f64()).unwrap_or(0.0) > 0.0 {
    return None;
  }
  let shapes = layer.get("shapes").and_then(|s| s.as_array())?;
  let mut paints: Vec<Paint> = Vec::new();
  let mut geoms: Vec<Geom> = Vec::new();
  walk_items(shapes, Mat::identity(), 1.0, &mut paints, &mut geoms, frame);
  if geoms.is_empty() {
    return None;
  }
  Some(Layer {
    geoms,
    paints,
    ks: layer.get("ks").cloned().unwrap_or(Value::Null),
    parent: None,
    ip: as_f32(layer.get("ip"), 0.0),
    op: as_f32(layer.get("op"), f32::MAX),
  })
}

fn compile_doc(doc: &Value, width: f32, height: f32) -> Result<Compiled, String> {
  let doc_w = as_f32(doc.get("w"), 512.0).max(1.0);
  let doc_h = as_f32(doc.get("h"), 512.0).max(1.0);
  let fr = as_f32(doc.get("fr"), 30.0).max(1.0);
  let ip = as_f32(doc.get("ip"), 0.0);
  let op = as_f32(doc.get("op"), ip + fr * 2.0);
  let raw_layers = doc
    .get("layers")
    .and_then(|l| l.as_array())
    .ok_or_else(|| "Lottie 缺少 layers 字段".to_string())?;

  // ind → 原始图层下标，用于解析 parent
  let mut ind_map: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
  for (i, layer) in raw_layers.iter().enumerate() {
    if let Some(ind) = layer.get("ind").and_then(|v| v.as_i64()) {
      ind_map.insert(ind, i);
    }
  }

  // 原始下标 → 编译后下标（被跳过的图层记为 None）
  let mut compiled_index_of_raw: Vec<Option<usize>> = Vec::with_capacity(raw_layers.len());
  let mut layers: Vec<Layer> = Vec::new();
  let mut raw_index_of_compiled: Vec<usize> = Vec::new();
  for (raw_idx, layer) in raw_layers.iter().enumerate() {
    match compile_layer(layer, ip) {
      Some(compiled) => {
        compiled_index_of_raw.push(Some(layers.len()));
        raw_index_of_compiled.push(raw_idx);
        layers.push(compiled);
      }
      None => compiled_index_of_raw.push(None),
    }
  }

  // 将 parent 的 ind 映射到编译后的图层下标
  for i in 0..layers.len() {
    let parent_ind = raw_layers[raw_index_of_compiled[i]]
      .get("parent")
      .and_then(|v| v.as_i64());
    if let Some(pid) = parent_ind {
      if let Some(raw_idx) = ind_map.get(&pid).copied() {
        if let Some(Some(compiled_idx)) = compiled_index_of_raw.get(raw_idx) {
          if *compiled_idx != i {
            layers[i].parent = Some(*compiled_idx);
          }
        }
      }
    }
  }

  if layers.is_empty() {
    return Err("没有可渲染的形状图层".to_string());
  }

  Ok(Compiled {
    doc_w,
    doc_h,
    fr,
    ip,
    op,
    width,
    height,
    layers,
  })
}

// ---------------------------------------------------------------------------
// 每帧渲染计划
// ---------------------------------------------------------------------------

struct Batch {
  kind: u32,
  first: usize,
  count: usize,
  color: [f32; 4],
  alpha: f32,
  p0: [f32; 2],
  p1: [f32; 2],
  stops: Vec<[f32; 5]>,
}

fn layer_world_matrix(c: &Compiled, index: usize, frame: f32, memo: &mut Vec<Option<Mat>>) -> Mat {
  if let Some(m) = memo[index] {
    return m;
  }
  let local = tr_matrix(&c.layers[index].ks, frame);
  let world = match c.layers[index].parent {
    Some(p) if p < c.layers.len() && p != index => layer_world_matrix(c, p, frame, memo).mul(&local),
    _ => local,
  };
  memo[index] = Some(world);
  world
}

fn render_frame(c: &Compiled, frame: f32) -> Vec<u8> {
  let sx = c.width / c.doc_w;
  let sy = c.height / c.doc_h;
  let mut batches: Vec<Batch> = Vec::new();
  let mut vertices: Vec<[f32; 2]> = Vec::new();
  let mut memo: Vec<Option<Mat>> = vec![None; c.layers.len()];

  // Lottie 的图层数组是「上层在前」，因此倒序绘制，保证上层后合成
  for i in (0..c.layers.len()).rev() {
    let layer = &c.layers[i];
    if frame < layer.ip || frame >= layer.op {
      continue;
    }
    let world = layer_world_matrix(c, i, frame, &mut memo);
    let layer_alpha = prop_f32(layer.ks.get("o"), frame, 100.0) / 100.0;
    if layer_alpha <= 0.001 {
      continue;
    }
    for geom in &layer.geoms {
      let paint = match layer.paints.get(geom.paint) {
        Some(p) => p,
        None => continue,
      };
      let first = vertices.len();
      for p in &geom.verts {
        let w = world.apply(*p);
        vertices.push([w[0] * sx, w[1] * sy]);
      }
      let g0 = world.apply(paint.p0);
      let g1 = world.apply(paint.p1);
      batches.push(Batch {
        kind: paint.kind,
        first,
        count: geom.verts.len(),
        color: paint.color,
        alpha: paint.alpha * layer_alpha,
        p0: [g0[0] * sx, g0[1] * sy],
        p1: [g1[0] * sx, g1[1] * sy],
        stops: paint.stops.clone(),
      });
    }
  }

  let mut words: Vec<u32> = Vec::with_capacity(1 + batches.len() * BATCH_WORDS + vertices.len() * 2);
  words.push(batches.len() as u32);
  for b in &batches {
    words.push(b.kind);
    words.push(b.first as u32);
    words.push(b.count as u32);
    words.push(b.color[0].to_bits());
    words.push(b.color[1].to_bits());
    words.push(b.color[2].to_bits());
    words.push(b.color[3].to_bits());
    words.push(b.alpha.to_bits());
    words.push(b.p0[0].to_bits());
    words.push(b.p0[1].to_bits());
    words.push(b.p1[0].to_bits());
    words.push(b.p1[1].to_bits());
    words.push(b.stops.len() as u32);
    for i in 0..MAX_STOPS {
      if let Some(s) = b.stops.get(i) {
        words.push(s[0].to_bits());
        words.push(s[1].to_bits());
        words.push(s[2].to_bits());
        words.push(s[3].to_bits());
        words.push(s[4].to_bits());
      } else {
        for _ in 0..5 {
          words.push(0);
        }
      }
    }
  }
  for v in &vertices {
    words.push(v[0].to_bits());
    words.push(v[1].to_bits());
  }

  let mut bytes: Vec<u8> = Vec::with_capacity(words.len() * 4);
  for w in words {
    bytes.extend_from_slice(&w.to_le_bytes());
  }
  bytes
}

// ---------------------------------------------------------------------------
// 静态状态与 WASM 导出
// ---------------------------------------------------------------------------

static mut COMPILED: Option<Compiled> = None;
static mut OUTPUT: Vec<u8> = Vec::new();
static mut LAST_ERROR: String = String::new();

fn set_error(message: &str) {
  unsafe {
    LAST_ERROR = message.to_string();
  }
}

#[no_mangle]
pub extern "C" fn l2s_alloc(len: usize) -> *mut u8 {
  let mut buf = vec![0u8; len];
  let ptr = buf.as_mut_ptr();
  std::mem::forget(buf);
  ptr
}

#[no_mangle]
pub extern "C" fn l2s_free(ptr: *mut u8, len: usize) {
  if !ptr.is_null() {
    unsafe {
      drop(Vec::from_raw_parts(ptr, len, len));
    }
  }
}

#[no_mangle]
pub extern "C" fn l2s_compile(ptr: *const u8, len: usize, width: f32, height: f32) -> u32 {
  let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
  let text = match std::str::from_utf8(bytes) {
    Ok(t) => t,
    Err(_) => {
      set_error("Lottie 数据不是合法 UTF-8");
      return 0;
    }
  };
  let doc: Value = match serde_json::from_str(text) {
    Ok(v) => v,
    Err(e) => {
      set_error(&format!("Lottie JSON 解析失败：{e}"));
      return 0;
    }
  };
  match compile_doc(&doc, width, height) {
    Ok(compiled) => {
      unsafe {
        COMPILED = Some(compiled);
        OUTPUT = Vec::new();
      }
      1
    }
    Err(e) => {
      set_error(&e);
      0
    }
  }
}

#[no_mangle]
pub extern "C" fn l2s_set_size(width: f32, height: f32) {
  unsafe {
    if let Some(c) = COMPILED.as_mut() {
      c.width = width.max(1.0);
      c.height = height.max(1.0);
    }
  }
}

#[no_mangle]
pub extern "C" fn l2s_render(frame: f32) -> u32 {
  unsafe {
    match COMPILED.as_ref() {
      Some(c) => {
        OUTPUT = render_frame(c, frame);
        1
      }
      None => {
        set_error("尚未编译 Lottie");
        0
      }
    }
  }
}

#[no_mangle]
pub extern "C" fn l2s_result_ptr() -> *const u8 {
  unsafe { OUTPUT.as_ptr() }
}

#[no_mangle]
pub extern "C" fn l2s_result_len() -> usize {
  unsafe { OUTPUT.len() }
}

#[no_mangle]
pub extern "C" fn l2s_error_ptr() -> *const u8 {
  unsafe { LAST_ERROR.as_ptr() }
}

#[no_mangle]
pub extern "C" fn l2s_error_len() -> usize {
  unsafe { LAST_ERROR.len() }
}

#[no_mangle]
pub extern "C" fn l2s_fps() -> f32 {
  unsafe { COMPILED.as_ref().map(|c| c.fr).unwrap_or(30.0) }
}

#[no_mangle]
pub extern "C" fn l2s_ip() -> f32 {
  unsafe { COMPILED.as_ref().map(|c| c.ip).unwrap_or(0.0) }
}

#[no_mangle]
pub extern "C" fn l2s_op() -> f32 {
  unsafe { COMPILED.as_ref().map(|c| c.op).unwrap_or(0.0) }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn parses_and_renders_simple_rect() {
    let json = serde_json::json!({
      "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
      "layers": [{
        "ddd": 0, "ind": 1, "ty": 4, "nm": "rect", "sr": 1,
        "ks": {
          "o": {"a": 0, "k": 100},
          "r": {"a": 0, "k": 0},
          "p": {"a": 0, "k": [200, 200, 0]},
          "a": {"a": 0, "k": [0, 0, 0]},
          "s": {"a": 0, "k": [100, 100, 100]}
        },
        "ao": 0,
        "shapes": [{
          "ty": "gr",
          "it": [
            {"ty": "rc", "p": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [200, 120]}, "r": {"a": 0, "k": 0}},
            {"ty": "fl", "c": {"a": 0, "k": [0.2, 0.6, 1, 1]}, "o": {"a": 0, "k": 100}},
            {"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}}
          ],
          "nm": "g"
        }],
        "ip": 0, "op": 60, "st": 0, "bm": 0
      }]
    });
    let compiled = compile_doc(&json, 512.0, 512.0).expect("compile");
    let bytes = render_frame(&compiled, 0.0);
    assert!(bytes.len() > 16);
    let batch_count = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert!(batch_count >= 1, "至少应产出一个绘制批次");
    // 顶点数量应大于 0（三角化成功）
    let vertex_words = (bytes.len() / 4) - 1 - (batch_count as usize) * BATCH_WORDS;
    assert!(vertex_words >= 6);
  }

  #[test]
  fn transform_rotates_around_anchor_then_translates() {
    // 旋转必须以锚点为轴、最后再平移到 p；顺序写反会绕原点旋转
    let tr = serde_json::json!({
      "p": {"a": 0, "k": [100, 0]},
      "a": {"a": 0, "k": [0, 0]},
      "s": {"a": 0, "k": [100, 100]},
      "r": {"a": 0, "k": 90}
    });
    let mat = tr_matrix(&tr, 0.0);
    let p = mat.apply([10.0, 0.0]);
    assert!((p[0] - 100.0).abs() < 0.01, "x={}", p[0]);
    assert!((p[1] - 10.0).abs() < 0.01, "y={}", p[1]);
  }

  #[test]
  fn ellipse_fill_area_matches_circle() {
    let json = serde_json::json!({
      "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
      "layers": [{
        "ddd": 0, "ind": 1, "ty": 4, "nm": "circle", "sr": 1,
        "ks": {
          "o": {"a": 0, "k": 100},
          "r": {"a": 0, "k": 0},
          "p": {"a": 0, "k": [200, 200, 0]},
          "a": {"a": 0, "k": [0, 0, 0]},
          "s": {"a": 0, "k": [100, 100, 100]}
        },
        "ao": 0,
        "shapes": [{
          "ty": "gr",
          "it": [
            {"ty": "el", "p": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [200, 200]}},
            {"ty": "fl", "c": {"a": 0, "k": [1, 0, 0, 1]}, "o": {"a": 0, "k": 100}},
            {"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}}
          ]
        }],
        "ip": 0, "op": 60, "st": 0, "bm": 0
      }]
    });
    let compiled = compile_doc(&json, 400.0, 400.0).expect("compile");
    let bytes = render_frame(&compiled, 0.0);
    let words: Vec<u32> = bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let batch_count = words[0] as usize;
    // 所有批次头之后才是共享顶点数组，批次头里的 vstart 是全局顶点下标
    let vertex_base = 1 + batch_count * BATCH_WORDS;
    let mut area = 0.0f32;
    for b in 0..batch_count {
      let header = 1 + b * BATCH_WORDS;
      let vertex_start = words[header + 1] as usize;
      let vertex_count = words[header + 2] as usize;
      for tri in (0..vertex_count).step_by(3) {
        let read = |i: usize| {
          let at = vertex_base + (vertex_start + i) * 2;
          [f32::from_bits(words[at]), f32::from_bits(words[at + 1])]
        };
        let a = read(tri);
        let b_pt = read(tri + 1);
        let c = read(tri + 2);
        area += (((b_pt[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b_pt[1] - a[1])).abs()) * 0.5;
      }
    }
    let expected = std::f32::consts::PI * 100.0 * 100.0;
    let ratio = area / expected;
    assert!(ratio > 0.97 && ratio < 1.03, "椭圆面积比 {ratio}");
  }
}
