//! lottie2shader —— 把 Lottie 动画编译成 GPU 三角面片渲染计划。
//!
//! 设计目标：
//! - 不依赖 wasm-bindgen：以裸 WASM 导出函数 + 线性内存交换数据，前端直接
//!   `WebAssembly.instantiate` 即可加载；
//! - 解析 Lottie JSON，在 Rust 侧完成路径求值、贝塞尔曲线离散化与三角化，
//!   几何一次性烘焙到图层局部坐标；动画中的形状属性变化则逐帧重新烘焙；
//! - 每帧只做「矩阵变换 + 顶点输出」，由 WebGL2 着色器完成光栅化、渐变、图片采样
//!   与轨道遮罩合成，从而把 Lottie 的绘制负担从 Canvas2D 移到 GPU；
//! - 以二进制批次格式输出：批次头（画笔类型/颜色/渐变/顶点区间/遮罩分组）+ 顶点坐标，
//!   前端按批次 drawArrays，避免逐帧解析 JSON。
//!
//! 支持范围：
//! - 形状图层（ty=4）、纯色层（ty=1）、图片层（ty=2）、空图层（ty=3）、预合成（ty=0）；
//! - 矩形 rc、椭圆 el、贝塞尔路径 sh、组 gr；
//! - 纯色填充 fl、纯色描边 st、线性/径向渐变填充 gf、渐变描边 gs；
//! - trim path（tm，支持按组合长度或按单条路径裁剪，支持动画）；
//! - 轨道遮罩（tt=1/2，alpha 与反相 alpha）；亮度遮罩不支持；
//! - 3D 图层的 rx/ry/rz 旋转与透视投影（不含 3D 相机）；
//! - 图层与组的变换、不透明度关键帧、父子关系、预合成嵌套。
//!
//! 明确不支持的特性会直接返回错误，由前端注册中心回退到 canvas，绝不静默画错：
//! 文本图层（需要字体轮廓）、表达式（需要 JS 引擎）、图层蒙版、图层特效、
//! 3D 相机、亮度轨道遮罩、星形/重复器/合并路径等形状修饰器。

use std::collections::{HashMap, HashSet};

use lyon::algorithms::measure::{PathMeasurements, SampleType};
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
/// 批次头固定占用字（u32）数，必须与前端 BATCH_WORDS 一致
const BATCH_WORDS: usize = 64;
/// 贝塞尔离散化的世界坐标容差（远小于 1 像素，保证矢量边缘足够平滑）
const TOLERANCE: f32 = 0.15;
/// 预合成最大嵌套深度，超过即为资源循环或异常文件
const MAX_DEPTH: usize = 8;

/// 图层类型（内部枚举）
const KIND_PRECOMP: u32 = 0;
const KIND_SOLID: u32 = 1;
const KIND_IMAGE: u32 = 2;
const KIND_NULL: u32 = 3;
const KIND_SHAPE: u32 = 4;

// ---------------------------------------------------------------------------
// 投影矩阵（3×3 齐次，可表达 2D 仿射与 3D 透视投影）
// ---------------------------------------------------------------------------

/// 3×3 齐次矩阵（行主序），uv 坐标为 (x, y, 1)：
/// x' = (m0*x + m1*y + m2) / w，y' = (m3*x + m4*y + m5) / w，w = m6*x + m7*y + m8。
/// 2D 仿射是第三行为 (0, 0, 1) 的特例；3D 透视把 w 写成线性函数即可在 Rust 侧做除法。
#[derive(Clone, Copy)]
struct Mat([f32; 9]);

impl Mat {
  fn identity() -> Self {
    Mat([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
  }

  fn translate(x: f32, y: f32) -> Self {
    Mat([1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0])
  }

  fn scale(x: f32, y: f32) -> Self {
    Mat([x, 0.0, 0.0, 0.0, y, 0.0, 0.0, 0.0, 1.0])
  }

  fn rotate(rad: f32) -> Self {
    let (s, c) = rad.sin_cos();
    Mat([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0])
  }

  /// 矩阵乘法：标准复合 `self ∘ other`（先应用 other，再应用 self）
  fn mul(&self, other: &Mat) -> Mat {
    let a = &self.0;
    let b = &other.0;
    let mut out = [0.0f32; 9];
    for row in 0..3 {
      for col in 0..3 {
        let mut sum = 0.0;
        for k in 0..3 {
          sum += a[row * 3 + k] * b[k * 3 + col];
        }
        out[row * 3 + col] = sum;
      }
    }
    Mat(out)
  }

  fn apply(&self, p: [f32; 2]) -> [f32; 2] {
    let m = &self.0;
    let x = m[0] * p[0] + m[1] * p[1] + m[2];
    let y = m[3] * p[0] + m[4] * p[1] + m[5];
    let mut w = m[6] * p[0] + m[7] * p[1] + m[8];
    if w.abs() < 1.0e-6 {
      w = if w < 0.0 { -1.0e-6 } else { 1.0e-6 };
    }
    [x / w, y / w]
  }

  /// 2D 部分的行列式开方，用于描边宽度随变换缩放
  fn scale_factor(&self) -> f32 {
    (self.0[0] * self.0[4] - self.0[1] * self.0[3]).abs().sqrt().max(0.0001)
  }

  fn is_identity(&self) -> bool {
    self.0[0] == 1.0
      && self.0[1] == 0.0
      && self.0[2] == 0.0
      && self.0[3] == 0.0
      && self.0[4] == 1.0
      && self.0[5] == 0.0
      && self.0[6] == 0.0
      && self.0[7] == 0.0
      && self.0[8] == 1.0
  }
}

/// 4×4 矩阵（行主序，列向量约定：out = M * v），仅用于构造 3D 图层变换
#[derive(Clone, Copy)]
struct Mat4([f32; 16]);

impl Mat4 {
  fn identity() -> Self {
    let mut m = [0.0f32; 16];
    m[0] = 1.0;
    m[5] = 1.0;
    m[10] = 1.0;
    m[15] = 1.0;
    Mat4(m)
  }

  fn mul(&self, other: &Mat4) -> Mat4 {
    let mut out = [0.0f32; 16];
    for row in 0..4 {
      for col in 0..4 {
        let mut sum = 0.0;
        for k in 0..4 {
          sum += self.0[row * 4 + k] * other.0[k * 4 + col];
        }
        out[row * 4 + col] = sum;
      }
    }
    Mat4(out)
  }

  fn translate(x: f32, y: f32, z: f32) -> Self {
    let mut m = Mat4::identity();
    m.0[3] = x;
    m.0[7] = y;
    m.0[11] = z;
    m
  }

  fn scale(x: f32, y: f32, z: f32) -> Self {
    let mut m = Mat4::identity();
    m.0[0] = x;
    m.0[5] = y;
    m.0[10] = z;
    m
  }

  fn rotate_x(rad: f32) -> Self {
    let (s, c) = rad.sin_cos();
    Mat4([1.0, 0.0, 0.0, 0.0, 0.0, c, -s, 0.0, 0.0, s, c, 0.0, 0.0, 0.0, 0.0, 1.0])
  }

  fn rotate_y(rad: f32) -> Self {
    let (s, c) = rad.sin_cos();
    Mat4([c, 0.0, s, 0.0, 0.0, 1.0, 0.0, 0.0, -s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0])
  }

  fn rotate_z(rad: f32) -> Self {
    let (s, c) = rad.sin_cos();
    Mat4([c, -s, 0.0, 0.0, s, c, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
  }

  /// 以透视距离 pe 投影到 z=0 平面（相机位于 z=-pe）
  fn project(&self, pe: f32) -> Mat {
    let pe = if pe > 0.0 { pe } else { 800.0 };
    Mat([
      pe * self.0[0],
      pe * self.0[1],
      pe * self.0[2],
      pe * self.0[4],
      pe * self.0[5],
      pe * self.0[6],
      self.0[8],
      self.0[9],
      pe + self.0[10],
    ])
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
      let end = value_vec(a.get("e").or_else(|| b.get("s")), dim, &start);
      return (0..dim)
        .map(|j| {
          start.get(j).copied().unwrap_or(0.0)
            + (end.get(j).copied().unwrap_or(0.0) - start.get(j).copied().unwrap_or(0.0)) * eased
        })
        .collect();
    }
  }
  value_vec(
    kfs.last().and_then(|kf| kf.get("s").or_else(|| kf.get("e"))),
    dim,
    def,
  )
}

fn prop_f32(prop: Option<&Value>, frame: f32, def: f32) -> f32 {
  prop_vec(prop, frame, 1, &[def])
    .first()
    .copied()
    .unwrap_or(def)
}

/// 求值一个「数值数组」属性（渐变停靠点等），长度由数据决定而非固定维度
fn prop_f32_array(prop: Option<&Value>, frame: f32) -> Vec<f32> {
  let Some(p) = prop else {
    return Vec::new();
  };
  let k = p.get("k");
  if !is_animated(Some(p)) {
    return match k {
      Some(Value::Array(arr)) => arr.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect(),
      _ => Vec::new(),
    };
  }
  let Some(kfs) = k.and_then(|k| k.as_array()) else {
    return Vec::new();
  };
  let parse = |v: Option<&Value>| -> Vec<f32> {
    v.and_then(|x| x.as_array())
      .map(|arr| arr.iter().map(|n| n.as_f64().unwrap_or(0.0) as f32).collect())
      .unwrap_or_default()
  };
  if kfs.is_empty() {
    return Vec::new();
  }
  let time = frame;
  let first = &kfs[0];
  if time <= as_f32(first.get("t"), 0.0) {
    return parse(first.get("s"));
  }
  for i in 0..kfs.len() - 1 {
    let a = &kfs[i];
    let b = &kfs[i + 1];
    let ta = as_f32(a.get("t"), 0.0);
    let tb = as_f32(b.get("t"), ta + 1.0);
    if time >= ta && time < tb {
      if a.get("h").and_then(|h| h.as_bool()).unwrap_or(false) {
        return parse(a.get("s"));
      }
      let raw = if tb > ta { (time - ta) / (tb - ta) } else { 0.0 };
      let (x1, y1) = easing_pair(a.get("o"));
      let (x2, y2) = easing_pair(b.get("i"));
      let eased = if x1 == 0.0 && y1 == 0.0 && x2 == 1.0 && y2 == 1.0 {
        raw
      } else {
        bezier_y(raw, x1, y1, x2, y2)
      };
      let start = parse(a.get("s"));
      let end = if a.get("e").is_some() {
        parse(a.get("e"))
      } else {
        parse(b.get("s"))
      };
      let n = start.len().max(end.len());
      return (0..n)
        .map(|j| {
          let s = start.get(j).copied().unwrap_or(0.0);
          let e = end.get(j).copied().unwrap_or(s);
          s + (e - s) * eased
        })
        .collect();
    }
  }
  parse(kfs.last().and_then(|kf| kf.get("s").or_else(|| kf.get("e"))))
}

/// 把图层 / 组变换解算成 2D 矩阵（锚点 → 缩放 → 旋转 → 位移）
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

/// 图层变换：3D 图层用完整 3D 变换 + 透视投影，其余走 2D 仿射
fn layer_matrix(ks: &Value, frame: f32, ddd: bool, pe: f32) -> Mat {
  if !ddd {
    return tr_matrix(ks, frame);
  }
  let p = prop_vec(ks.get("p"), frame, 3, &[0.0, 0.0, 0.0]);
  let a = prop_vec(ks.get("a"), frame, 3, &[0.0, 0.0, 0.0]);
  let s = prop_vec(ks.get("s"), frame, 3, &[100.0, 100.0, 100.0]);
  let rz = prop_f32(ks.get("rz"), frame, prop_f32(ks.get("r"), frame, 0.0)).to_radians();
  let ry = prop_f32(ks.get("ry"), frame, 0.0).to_radians();
  let rx = prop_f32(ks.get("rx"), frame, 0.0).to_radians();
  // Lottie / AE 的欧拉顺序：先 Z，再 Y，最后 X
  let m = Mat4::translate(p[0], p[1], p[2])
    .mul(&Mat4::rotate_z(rz))
    .mul(&Mat4::rotate_y(ry))
    .mul(&Mat4::rotate_x(rx))
    .mul(&Mat4::scale(s[0] / 100.0, s[1] / 100.0, s[2] / 100.0))
    .mul(&Mat4::translate(-a[0], -a[1], -a[2]));
  m.project(pe)
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

/// 一段已烘焙到图层局部坐标的几何数据；图片批次额外带 UV
struct Geom {
  verts: Vec<[f32; 2]>,
  paint: usize,
  uvs: Vec<[f32; 2]>,
  image: Option<usize>,
}

/// 一个编译后的图层。预合成通过 children + container 形成树，
/// 父子关系（parent）与容器关系（container）都通过下标表达，渲染时按需解算。
struct Layer {
  kind: u32,
  geoms: Vec<Geom>,
  paints: Vec<Paint>,
  /// 含动画形状属性的图层需要逐帧重新烘焙（保存原始 shapes）
  shapes: Option<Value>,
  dynamic: bool,
  ks: Value,
  ddd: bool,
  pe: f32,
  ip: f32,
  op: f32,
  /// 所属时间轴的起点（预合成的 st），用于把全局帧换算成本地帧
  time_offset: f32,
  parent: Option<usize>,
  container: Option<usize>,
  children: Vec<usize>,
  /// 轨道遮罩分组：0 表示不参与遮罩
  group: u32,
  /// 0 普通、1 遮罩源、2 被遮罩
  role: u32,
  /// 1 alpha、2 反相 alpha
  matte_mode: u32,
}

struct ImageAsset {
  src: String,
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
  roots: Vec<usize>,
  images: Vec<ImageAsset>,
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

/// 递归检测 JSON 中是否存在动画属性（`a` 为 true 或大于 0 的数字）。
/// 形状图层只要有任何属性在动，就需要逐帧重新烘焙；图层自身 ks 由渲染期矩阵处理，不在此列。
fn has_animation(value: &Value) -> bool {
  match value {
    Value::Object(map) => {
      let animated = match map.get("a") {
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(num)) => num.as_f64().unwrap_or(0.0) > 0.0,
        _ => false,
      };
      animated || map.values().any(has_animation)
    }
    Value::Array(arr) => arr.iter().any(has_animation),
    _ => false,
  }
}

/// 一条贝塞尔路径的顶点与控制点数据
struct PathData {
  verts: Vec<[f32; 2]>,
  ins: Vec<[f32; 2]>,
  outs: Vec<[f32; 2]>,
  closed: bool,
}

fn parse_path_data(value: &Value) -> Option<PathData> {
  // 关键帧路径的 `s` 是「路径对象数组」（如 `[{c,i,o,v}]`），静态路径则直接是对象
  let data = if value.is_array() {
    value.as_array()?.first()?.clone()
  } else {
    value.clone()
  };
  let verts = arr2(data.get("v"));
  if verts.is_empty() {
    return None;
  }
  Some(PathData {
    verts,
    ins: arr2(data.get("i")),
    outs: arr2(data.get("o")),
    closed: data.get("c").and_then(|c| c.as_bool()).unwrap_or(false),
  })
}

fn lerp_points(a: &[[f32; 2]], b: &[[f32; 2]], k: f32) -> Vec<[f32; 2]> {
  (0..a.len())
    .map(|i| {
      let target = b.get(i).copied().unwrap_or(a[i]);
      [
        a[i][0] + (target[0] - a[i][0]) * k,
        a[i][1] + (target[1] - a[i][1]) * k,
      ]
    })
    .collect()
}

/// 求值路径属性；支持顶点数一致的形状形变（path morph）插值。
/// 顶点数不一致时无法做拓扑合并，退化为最接近的一侧，避免出现空白帧。
fn path_data_at(prop: &Value, frame: f32) -> Option<PathData> {
  if !is_animated(Some(prop)) {
    return parse_path_data(prop.get("k")?);
  }
  let keyframes = prop.get("k")?.as_array()?;
  if keyframes.is_empty() {
    return None;
  }
  let first = keyframes.first()?;
  if frame <= as_f32(first.get("t"), 0.0) {
    return parse_path_data(first.get("s")?);
  }
  for i in 0..keyframes.len() - 1 {
    let a = &keyframes[i];
    let b = &keyframes[i + 1];
    let ta = as_f32(a.get("t"), 0.0);
    let tb = as_f32(b.get("t"), ta + 1.0);
    if frame >= ta && frame < tb {
      if a.get("h").and_then(|h| h.as_bool()).unwrap_or(false) {
        return parse_path_data(a.get("s")?);
      }
      let start = parse_path_data(a.get("s")?)?;
      let end = match a.get("e") {
        Some(e) => parse_path_data(e)?,
        None => parse_path_data(b.get("s")?)?,
      };
      let raw = if tb > ta { (frame - ta) / (tb - ta) } else { 0.0 };
      let (x1, y1) = easing_pair(a.get("o"));
      let (x2, y2) = easing_pair(b.get("i"));
      let t = if x1 == 0.0 && y1 == 0.0 && x2 == 1.0 && y2 == 1.0 {
        raw
      } else {
        bezier_y(raw, x1, y1, x2, y2)
      };
      if start.verts.len() == end.verts.len()
        && start.ins.len() == end.ins.len()
        && start.outs.len() == end.outs.len()
      {
        return Some(PathData {
          verts: lerp_points(&start.verts, &end.verts, t),
          ins: lerp_points(&start.ins, &end.ins, t),
          outs: lerp_points(&start.outs, &end.outs, t),
          closed: start.closed,
        });
      }
      return Some(if t < 0.5 { start } else { end });
    }
  }
  let last = keyframes.last()?;
  parse_path_data(last.get("s").or_else(|| last.get("e"))?)
}

fn path_from_shape(item: &Value, mat: &Mat, frame: f32) -> Option<Path> {
  let data = path_data_at(item.get("ks")?, frame)?;
  bezier_path(&data.verts, &data.ins, &data.outs, data.closed, mat)
}

fn path_from_ellipse(item: &Value, mat: &Mat, frame: f32) -> Option<Path> {
  let p = prop_vec(item.get("p"), frame, 2, &[0.0, 0.0]);
  let s = prop_vec(item.get("s"), frame, 2, &[0.0, 0.0]);
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

fn path_from_rect(item: &Value, mat: &Mat, frame: f32) -> Option<Path> {
  let p = prop_vec(item.get("p"), frame, 2, &[0.0, 0.0]);
  let s = prop_vec(item.get("s"), frame, 2, &[0.0, 0.0]);
  let r = prop_f32(item.get("r"), frame, 0.0).max(0.0);
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
// trim path
// ---------------------------------------------------------------------------

fn path_length(path: &Path) -> f32 {
  let measurements = PathMeasurements::from_path(path, TOLERANCE);
  let sampler = measurements.create_sampler(path, SampleType::Distance);
  sampler.length()
}

/// 把归一化区间 [s, e]（可带旋转偏移 o）转成 0~1 的多段区间；闭环时支持环绕
fn trim_ranges(s: f32, e: f32, o: f32, closed: bool) -> Vec<(f32, f32)> {
  let start = s + o;
  let end = e + o;
  if !closed {
    let a = start.min(end).clamp(0.0, 1.0);
    let b = start.max(end).clamp(0.0, 1.0);
    return if b > a { vec![(a, b)] } else { Vec::new() };
  }
  if end <= start {
    return Vec::new();
  }
  if end - start >= 1.0 {
    return vec![(0.0, 1.0)];
  }
  // 沿长度方向逐圈切分，保证每段都落在 [0, 1) 内
  let mut ranges = Vec::new();
  let mut cursor = start;
  while cursor < end {
    let base = cursor.floor();
    let seg_end = (base + 1.0).min(end);
    let a = cursor - base;
    let b = seg_end - base;
    if b > a + 1.0e-6 {
      ranges.push((a.min(1.0), b.min(1.0)));
    }
    cursor = seg_end;
  }
  ranges
}

/// 按归一化区间裁一段路径
fn trim_path(path: &Path, ranges: &[(f32, f32)]) -> Option<Path> {
  if ranges.is_empty() {
    return None;
  }
  let measurements = PathMeasurements::from_path(path, TOLERANCE);
  let mut sampler = measurements.create_sampler(path, SampleType::Distance);
  let len = sampler.length();
  if len <= 0.0 {
    return None;
  }
  let mut builder = Path::builder();
  let mut any = false;
  for &(a, b) in ranges {
    let start = a.max(0.0).min(1.0) * len;
    let end = b.max(0.0).min(1.0) * len;
    if end <= start {
      continue;
    }
    sampler.split_range(start..end, &mut builder);
    any = true;
  }
  if any {
    Some(builder.build())
  } else {
    None
  }
}

/// trim path 主逻辑：m=1 时按所有路径的组合长度裁剪，m=2 时每条路径各自裁剪
fn apply_trim(paths: &[Path], s: f32, e: f32, o: f32, individually: bool) -> Vec<Path> {
  if paths.is_empty() {
    return Vec::new();
  }
  if individually {
    let ranges = trim_ranges(s, e, o, true);
    if ranges.is_empty() {
      return Vec::new();
    }
    return paths.iter().filter_map(|p| trim_path(p, &ranges)).collect();
  }
  let lengths: Vec<f32> = paths.iter().map(|p| path_length(p)).collect();
  let total: f32 = lengths.iter().sum();
  if total <= 0.0 {
    return Vec::new();
  }
  let ranges = trim_ranges(s, e, o, true);
  if ranges.is_empty() {
    return Vec::new();
  }
  let mut out = Vec::new();
  let mut cum = 0.0f32;
  for (path, len) in paths.iter().zip(lengths.iter()) {
    let frac = len / total;
    if frac <= 1.0e-6 {
      cum += frac;
      continue;
    }
    let local: Vec<(f32, f32)> = ranges
      .iter()
      .filter_map(|&(a, b)| {
        let la = ((a - cum) / frac).max(0.0);
        let lb = ((b - cum) / frac).min(1.0);
        if lb > la + 1.0e-6 {
          Some((la, lb))
        } else {
          None
        }
      })
      .collect();
    if let Some(trimmed) = trim_path(path, &local) {
      out.push(trimmed);
    }
    cum += frac;
  }
  out
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
    geoms.push(Geom {
      verts,
      paint,
      uvs: Vec::new(),
      image: None,
    });
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
        if let Some(vertex) = geometry.vertices.get(idx as usize) {
          verts.push([vertex.x, vertex.y]);
        }
      }
    }
  }
  if !verts.is_empty() {
    geoms.push(Geom {
      verts,
      paint,
      uvs: Vec::new(),
      image: None,
    });
  }
}

// ---------------------------------------------------------------------------
// 画笔解析
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

/// 渐变画笔（gf 填充 / gs 描边共用）
fn gradient_paint(item: &Value, alpha: f32, frame: f32) -> Paint {
  let s = prop_vec(item.get("s"), frame, 2, &[0.0, 0.0]);
  let e = prop_vec(item.get("e"), frame, 2, &[1.0, 0.0]);
  // Lottie：t=1 线性，t=2 径向；部分导出用 r=1 表示径向
  let is_radial = as_f32(item.get("t"), 0.0) == 2.0 || as_f32(item.get("r"), 0.0) >= 1.0;
  let mut stops: Vec<[f32; 5]> = Vec::new();
  if let Some(g) = item.get("g") {
    let count = as_f32(g.get("p"), 0.0) as usize;
    let values = prop_f32_array(g.get("k"), frame);
    // 色标可能是 4 分量（rgb）或 5 分量（rgba），按总长度判断
    if !values.is_empty() {
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

fn parse_hex_color(text: &str) -> [f32; 4] {
  let hex = text.trim_start_matches('#');
  if hex.len() >= 6 {
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap_or(0) as f32 / 255.0;
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap_or(0) as f32 / 255.0;
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap_or(0) as f32 / 255.0;
    return [r, g, b, 1.0];
  }
  [0.0, 0.0, 0.0, 1.0]
}

// ---------------------------------------------------------------------------
// 形状组遍历
// ---------------------------------------------------------------------------

fn walk_items(
  items: &[Value],
  base: Mat,
  alpha: f32,
  frame: f32,
) -> (Vec<Paint>, Vec<Geom>) {
  let mut paints: Vec<Paint> = Vec::new();
  let mut geoms: Vec<Geom> = Vec::new();
  // 组内 `tr` 决定本组变换与不透明度
  let mut mat = base;
  let mut group_alpha = alpha;
  if let Some(tr) = items.iter().find(|i| i.get("ty").and_then(|t| t.as_str()) == Some("tr")) {
    mat = base.mul(&tr_matrix(tr, frame));
    group_alpha *= prop_f32(tr.get("o"), frame, 100.0) / 100.0;
  }

  // 组内已收集的路径，供后续填充 / 描边画笔使用；trim 会就地改写它们
  let mut paths: Vec<Path> = Vec::new();

  for item in items {
    let ty = item.get("ty").and_then(|t| t.as_str()).unwrap_or("");
    match ty {
      "gr" => {
        if let Some(child) = item.get("it").and_then(|i| i.as_array()) {
          let (child_paints, mut child_geoms) = walk_items(child, mat, group_alpha, frame);
          // 子组的画笔下标是局部编号，合并到父组后需要整体平移
          let offset = paints.len();
          for geom in &mut child_geoms {
            if geom.image.is_none() {
              geom.paint += offset;
            }
          }
          paints.extend(child_paints);
          geoms.extend(child_geoms);
        }
      }
      "sh" => {
        if let Some(p) = path_from_shape(item, &mat, frame) {
          paths.push(p);
        }
      }
      "el" => {
        if let Some(p) = path_from_ellipse(item, &mat, frame) {
          paths.push(p);
        }
      }
      "rc" => {
        if let Some(p) = path_from_rect(item, &mat, frame) {
          paths.push(p);
        }
      }
      "tm" => {
        let s = prop_f32(item.get("s"), frame, 0.0) / 100.0;
        let e = prop_f32(item.get("e"), frame, 100.0) / 100.0;
        let o = prop_f32(item.get("o"), frame, 0.0) / 360.0;
        // m=1 同时（组合长度），m=2 单独（每条路径各裁一段）
        let individually = as_f32(item.get("m"), 1.0) >= 2.0;
        paths = apply_trim(&paths, s, e, o, individually);
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
        tessellate_fill(&paths, rule, index, &mut geoms);
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
        tessellate_stroke(&paths, width, join, cap, index, &mut geoms);
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
        tessellate_fill(&paths, rule, index, &mut geoms);
      }
      "gs" => {
        let paint = gradient_paint(item, group_alpha, frame);
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
        tessellate_stroke(&paths, width, join, cap, index, &mut geoms);
      }
      _ => {}
    }
  }
  (paints, geoms)
}

// ---------------------------------------------------------------------------
// 支持性检查（不支持的特性必须显式报错，交给前端回退）
// ---------------------------------------------------------------------------

fn has_expression(value: &Value) -> bool {
  match value {
    Value::Object(map) => map
      .iter()
      .any(|(key, v)| (key == "x" && v.is_string()) || has_expression(v)),
    Value::Array(arr) => arr.iter().any(has_expression),
    _ => false,
  }
}

fn check_shapes(items: &[Value]) -> Result<(), String> {
  for item in items {
    let ty = item.get("ty").and_then(|t| t.as_str()).unwrap_or("");
    match ty {
      "gr" => {
        if let Some(children) = item.get("it").and_then(|i| i.as_array()) {
          check_shapes(children)?;
        }
      }
      "sh" | "el" | "rc" | "fl" | "st" | "gf" | "gs" | "tr" | "tm" | "no" => {}
      other => {
        return Err(format!("不支持的形状类型：{other}（请回退 canvas 渲染）"));
      }
    }
  }
  Ok(())
}

fn find_asset<'a>(doc: &'a Value, id: &str) -> Option<&'a Value> {
  doc.get("assets")
    .and_then(|a| a.as_array())
    .and_then(|arr| {
      arr
        .iter()
        .find(|x| x.get("id").and_then(|i| i.as_str()) == Some(id))
    })
}

fn check_layers(
  doc: &Value,
  layers: &[Value],
  depth: usize,
  seen: &mut HashSet<String>,
) -> Result<(), String> {
  if depth > MAX_DEPTH {
    return Err("预合成嵌套过深（可能存在资源循环）".to_string());
  }
  for layer in layers {
    if has_expression(layer) {
      return Err("检测到表达式（expression），需要 JS 引擎，暂不支持".to_string());
    }
    let ty = layer.get("ty").and_then(|t| t.as_f64()).unwrap_or(0.0) as i64;
    match ty {
      0 => {
        let ref_id = layer.get("refId").and_then(|v| v.as_str()).unwrap_or("");
        if !seen.insert(ref_id.to_string()) {
          continue;
        }
        let asset = find_asset(doc, ref_id)
          .ok_or_else(|| format!("预合成缺少资源：{ref_id}"))?;
        let sub = asset
          .get("layers")
          .and_then(|l| l.as_array())
          .ok_or_else(|| format!("预合成没有图层：{ref_id}"))?;
        check_layers(doc, sub, depth + 1, seen)?;
      }
      1 | 2 | 3 | 4 => {}
      5 => return Err("文本图层需要字体轮廓，暂不支持".to_string()),
      13 => return Err("3D 相机图层暂不支持".to_string()),
      other => return Err(format!("不支持的图层类型 ty={other}")),
    }
    if let Some(masks) = layer.get("masksProperties").and_then(|m| m.as_array()) {
      if !masks.is_empty() {
        return Err("图层蒙版（masksProperties）暂不支持".to_string());
      }
    }
    if let Some(effects) = layer.get("ef").and_then(|e| e.as_array()) {
      if !effects.is_empty() {
        return Err("图层特效（ef）暂不支持".to_string());
      }
    }
    let tt = layer.get("tt").and_then(|t| t.as_f64()).unwrap_or(0.0) as i64;
    if tt == 3 || tt == 4 {
      return Err("亮度轨道遮罩暂不支持".to_string());
    }
    if ty == 4 {
      if let Some(shapes) = layer.get("shapes").and_then(|s| s.as_array()) {
        check_shapes(shapes)?;
      }
    }
  }
  Ok(())
}

// ---------------------------------------------------------------------------
// 编译
// ---------------------------------------------------------------------------

struct CompileCtx<'a> {
  doc: &'a Value,
  images: Vec<ImageAsset>,
  image_map: HashMap<String, usize>,
  group_seq: u32,
}

impl<'a> CompileCtx<'a> {
  fn next_group(&mut self) -> u32 {
    self.group_seq += 1;
    self.group_seq
  }
}

fn quad_geometry(w: f32, h: f32, paint: usize) -> Geom {
  Geom {
    verts: vec![
      [0.0, 0.0],
      [w, 0.0],
      [w, h],
      [0.0, 0.0],
      [w, h],
      [0.0, h],
    ],
    paint,
    uvs: Vec::new(),
    image: None,
  }
}

fn image_geometry(w: f32, h: f32, image: usize) -> Geom {
  Geom {
    verts: vec![
      [0.0, 0.0],
      [w, 0.0],
      [w, h],
      [0.0, 0.0],
      [w, h],
      [0.0, h],
    ],
    paint: usize::MAX,
    uvs: vec![
      [0.0, 0.0],
      [1.0, 0.0],
      [1.0, 1.0],
      [0.0, 0.0],
      [1.0, 1.0],
      [0.0, 1.0],
    ],
    image: Some(image),
  }
}

fn image_source(asset: &Value) -> Result<String, String> {
  let p = asset.get("p").and_then(|v| v.as_str()).unwrap_or("");
  if p.starts_with("data:") {
    return Ok(p.to_string());
  }
  if p.is_empty() {
    return Err("图片资源缺少路径".to_string());
  }
  let u = asset.get("u").and_then(|v| v.as_str()).unwrap_or("");
  Ok(format!("{u}{p}"))
}

fn placeholder(
  container: Option<usize>,
  time_offset: f32,
  ip: f32,
  op: f32,
  ks: Value,
  ddd: bool,
  pe: f32,
) -> Layer {
  Layer {
    kind: KIND_NULL,
    geoms: Vec::new(),
    paints: Vec::new(),
    shapes: None,
    dynamic: false,
    ks,
    ddd,
    pe,
    ip,
    op,
    time_offset,
    parent: None,
    container,
    children: Vec::new(),
    group: 0,
    role: 0,
    matte_mode: 0,
  }
}

fn compile_scope(
  ctx: &mut CompileCtx,
  raw: &[Value],
  container: Option<usize>,
  time_offset: f32,
  out: &mut Vec<Layer>,
  depth: usize,
) -> Result<Vec<usize>, String> {
  if depth > MAX_DEPTH {
    return Err("预合成嵌套过深（可能存在资源循环）".to_string());
  }
  let doc = ctx.doc;
  let mut ind_map: HashMap<i64, usize> = HashMap::new();
  for (i, layer) in raw.iter().enumerate() {
    if let Some(ind) = layer.get("ind").and_then(|v| v.as_i64()) {
      ind_map.insert(ind, i);
    }
  }

  let mut entry_of_raw: Vec<Option<usize>> = vec![None; raw.len()];
  let mut entries: Vec<usize> = Vec::new();

  for (idx, layer) in raw.iter().enumerate() {
    if layer.get("hd").and_then(|h| h.as_bool()).unwrap_or(false) {
      continue;
    }
    let ty = layer.get("ty").and_then(|t| t.as_f64()).unwrap_or(0.0) as i64;
    if layer.get("td").and_then(|t| t.as_f64()).unwrap_or(0.0) > 0.0 && ty != 0 {
      // 遮罩源图层仍然要编译（它参与遮罩），这里不做跳过
    }
    let ks = layer.get("ks").cloned().unwrap_or(Value::Null);
    let ip = as_f32(layer.get("ip"), 0.0);
    let op = as_f32(layer.get("op"), f32::MAX);
    let ddd = layer.get("ddd").and_then(|d| d.as_f64()).unwrap_or(0.0) > 0.0;
    let pe = as_f32(layer.get("pe"), 0.0);

    let slot = out.len();
    out.push(placeholder(container, time_offset, ip, op, ks.clone(), ddd, pe));
    entry_of_raw[idx] = Some(slot);
    entries.push(slot);

    let mut entry = placeholder(container, time_offset, ip, op, ks, ddd, pe);
    match ty {
      4 => {
        let shapes = layer
          .get("shapes")
          .and_then(|s| s.as_array())
          .cloned()
          .unwrap_or_default();
        let dynamic = shapes.iter().any(has_animation);
        let (paints, geoms) = walk_items(&shapes, Mat::identity(), 1.0, ip);
        entry.geoms = geoms;
        entry.paints = paints;
        entry.dynamic = dynamic;
        if dynamic {
          entry.shapes = layer.get("shapes").cloned();
        }
        entry.kind = KIND_SHAPE;
      }
      1 => {
        let sw = as_f32(layer.get("sw"), 0.0).max(0.0);
        let sh = as_f32(layer.get("sh"), 0.0).max(0.0);
        let color = parse_hex_color(layer.get("sc").and_then(|s| s.as_str()).unwrap_or("#000000"));
        entry.paints.push(Paint {
          kind: 0,
          color,
          alpha: 1.0,
          p0: [0.0, 0.0],
          p1: [1.0, 1.0],
          stops: Vec::new(),
        });
        entry.geoms.push(quad_geometry(sw, sh, 0));
        entry.kind = KIND_SOLID;
      }
      2 => {
        let ref_id = layer.get("refId").and_then(|v| v.as_str()).unwrap_or("");
        let asset = find_asset(doc, ref_id)
          .ok_or_else(|| format!("图片图层缺少资源：{ref_id}"))?;
        let iw = as_f32(asset.get("w"), as_f32(layer.get("w"), 0.0));
        let ih = as_f32(asset.get("h"), as_f32(layer.get("h"), 0.0));
        if iw <= 0.0 || ih <= 0.0 {
          return Err(format!("图片资源尺寸无效：{ref_id}"));
        }
        let src = image_source(asset)?;
        let image = match ctx.image_map.get(ref_id) {
          Some(i) => *i,
          None => {
            let i = ctx.images.len();
            ctx.images.push(ImageAsset { src });
            ctx.image_map.insert(ref_id.to_string(), i);
            i
          }
        };
        entry.geoms.push(image_geometry(iw, ih, image));
        entry.kind = KIND_IMAGE;
      }
      0 => {
        let ref_id = layer.get("refId").and_then(|v| v.as_str()).unwrap_or("");
        let asset = find_asset(doc, ref_id)
          .ok_or_else(|| format!("预合成缺少资源：{ref_id}"))?;
        let sub = asset
          .get("layers")
          .and_then(|l| l.as_array())
          .ok_or_else(|| format!("预合成没有图层：{ref_id}"))?;
        let st = as_f32(layer.get("st"), 0.0);
        let children = compile_scope(
          ctx,
          sub,
          Some(slot),
          time_offset + st,
          out,
          depth + 1,
        )?;
        entry.children = children;
        entry.kind = KIND_PRECOMP;
      }
      _ => {
        entry.kind = KIND_NULL;
      }
    }
    out[slot] = entry;
  }

  // 解析父子关系（只在当前层列表内）
  for (idx, layer) in raw.iter().enumerate() {
    let Some(slot) = entry_of_raw[idx] else {
      continue;
    };
    let Some(parent_ind) = layer.get("parent").and_then(|v| v.as_i64()) else {
      continue;
    };
    let Some(raw_idx) = ind_map.get(&parent_ind).copied() else {
      continue;
    };
    let Some(parent_slot) = entry_of_raw[raw_idx] else {
      continue;
    };
    if parent_slot != slot {
      out[slot].parent = Some(parent_slot);
    }
  }

  // 轨道遮罩：tt 图层使用紧邻其上的 td 图层作为遮罩源
  for idx in 1..raw.len() {
    let tt = raw[idx].get("tt").and_then(|v| v.as_f64()).unwrap_or(0.0) as i64;
    if tt == 0 {
      continue;
    }
    if tt != 1 && tt != 2 {
      continue;
    }
    let Some(mask_slot) = entry_of_raw[idx] else {
      continue;
    };
    let Some(src_slot) = entry_of_raw[idx - 1] else {
      continue;
    };
    let td = raw[idx - 1].get("td").and_then(|v| v.as_f64()).unwrap_or(0.0);
    if td <= 0.0 {
      continue;
    }
    if out[src_slot].group != 0 || out[mask_slot].group != 0 {
      continue;
    }
    let gid = ctx.next_group();
    out[src_slot].group = gid;
    out[src_slot].role = 1;
    out[mask_slot].group = gid;
    out[mask_slot].role = 2;
    out[mask_slot].matte_mode = tt as u32;
  }

  Ok(entries)
}

fn compile_doc(doc: &Value, width: f32, height: f32) -> Result<Compiled, String> {
  check_layers(
    doc,
    doc.get("layers")
      .and_then(|l| l.as_array())
      .ok_or_else(|| "Lottie 缺少 layers 字段".to_string())?,
    0,
    &mut HashSet::new(),
  )?;

  let doc_w = as_f32(doc.get("w"), 512.0).max(1.0);
  let doc_h = as_f32(doc.get("h"), 512.0).max(1.0);
  let fr = as_f32(doc.get("fr"), 30.0).max(1.0);
  let ip = as_f32(doc.get("ip"), 0.0);
  let op = as_f32(doc.get("op"), ip + fr * 2.0);
  let raw_layers = doc
    .get("layers")
    .and_then(|l| l.as_array())
    .ok_or_else(|| "Lottie 缺少 layers 字段".to_string())?;

  let mut ctx = CompileCtx {
    doc,
    images: Vec::new(),
    image_map: HashMap::new(),
    group_seq: 0,
  };
  let mut layers: Vec<Layer> = Vec::new();
  let roots = compile_scope(&mut ctx, raw_layers, None, 0.0, &mut layers, 0)?;

  if !layers.iter().any(|l| !l.geoms.is_empty()) {
    return Err("没有可渲染的图层".to_string());
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
    roots,
    images: ctx.images,
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
  uv_first: usize,
  group: u32,
  matte: u32,
  role: u32,
  image: Option<usize>,
}

struct Renderer<'a> {
  c: &'a Compiled,
  frame: f32,
  world_memo: Vec<Option<Mat>>,
  active_memo: Vec<Option<bool>>,
  alpha_memo: Vec<Option<f32>>,
  visiting: Vec<bool>,
  active_visiting: Vec<bool>,
  emitted: HashSet<u32>,
  batches: Vec<Batch>,
  vertices: Vec<[f32; 2]>,
  uvs: Vec<[f32; 2]>,
}

impl<'a> Renderer<'a> {
  fn new(c: &'a Compiled, frame: f32) -> Self {
    let n = c.layers.len();
    Renderer {
      c,
      frame,
      world_memo: vec![None; n],
      active_memo: vec![None; n],
      alpha_memo: vec![None; n],
      visiting: vec![false; n],
      active_visiting: vec![false; n],
      emitted: HashSet::new(),
      batches: Vec::new(),
      vertices: Vec::new(),
      uvs: Vec::new(),
    }
  }

  fn world_of(&mut self, i: usize) -> Mat {
    if let Some(m) = self.world_memo[i] {
      return m;
    }
    if self.visiting[i] {
      return Mat::identity();
    }
    self.visiting[i] = true;
    let layer = &self.c.layers[i];
    let local = layer_matrix(&layer.ks, self.frame - layer.time_offset, layer.ddd, layer.pe);
    let base = if let Some(parent) = layer.parent {
      self.world_of(parent)
    } else if let Some(container) = layer.container {
      self.world_of(container)
    } else {
      Mat::identity()
    };
    let world = base.mul(&local);
    self.visiting[i] = false;
    self.world_memo[i] = Some(world);
    world
  }

  fn active(&mut self, i: usize) -> bool {
    if let Some(v) = self.active_memo[i] {
      return v;
    }
    if self.active_visiting[i] {
      return true;
    }
    self.active_visiting[i] = true;
    let layer = &self.c.layers[i];
    let local = self.frame - layer.time_offset;
    let own = local >= layer.ip && local < layer.op;
    let ok = own
      && match layer.container {
        Some(container) => self.active(container),
        None => true,
      };
    self.active_visiting[i] = false;
    self.active_memo[i] = Some(ok);
    ok
  }

  fn alpha_of(&mut self, i: usize) -> f32 {
    if let Some(a) = self.alpha_memo[i] {
      return a;
    }
    let layer = &self.c.layers[i];
    let local = self.frame - layer.time_offset;
    let own = prop_f32(layer.ks.get("o"), local, 100.0) / 100.0;
    let alpha = match layer.container {
      Some(container) => self.alpha_of(container) * own,
      None => own,
    };
    self.alpha_memo[i] = Some(alpha);
    alpha
  }

  fn render_list(&mut self, list: &[usize], group: u32, role: u32, matte: u32) {
    // 继承分组（预合成子图层）：整段按同一遮罩角色绘制
    if group != 0 {
      for &i in list.iter().rev() {
        if !self.active(i) {
          continue;
        }
        self.render_entry(i, group, role, matte);
      }
      return;
    }

    for &i in list.iter().rev() {
      let layer = &self.c.layers[i];
      if layer.role == 1 {
        // 遮罩源在对应的被遮罩图层处统一绘制，保证批次连续
        continue;
      }
      if !self.active(i) {
        continue;
      }
      if layer.role == 2 && layer.group != 0 {
        if !self.emitted.contains(&layer.group) {
          self.emitted.insert(layer.group);
          for &j in list.iter().rev() {
            let source = &self.c.layers[j];
            if source.role == 1 && source.group == layer.group {
              self.render_entry(j, layer.group, 1, 0);
            }
          }
        }
        let matte_mode = layer.matte_mode;
        self.render_entry(i, layer.group, 2, matte_mode);
      } else {
        self.render_entry(i, 0, 0, 0);
      }
    }
  }

  fn render_entry(&mut self, i: usize, group: u32, role: u32, matte: u32) {
    if !self.active(i) {
      return;
    }
    let world = self.world_of(i);
    let alpha = self.alpha_of(i);
    if alpha <= 0.001 {
      return;
    }
    let kind = self.c.layers[i].kind;
    if kind == KIND_PRECOMP {
      let children = self.c.layers[i].children.clone();
      self.render_list(&children, group, role, matte);
      return;
    }
    if kind == KIND_NULL {
      return;
    }
    self.emit(i, world, alpha, group, role, matte);
  }

  fn emit(&mut self, i: usize, world: Mat, layer_alpha: f32, group: u32, role: u32, matte: u32) {
    let c = self.c;
    let layer = &c.layers[i];
    let sx = c.width / c.doc_w;
    let sy = c.height / c.doc_h;

    let mut dyn_paints: Vec<Paint> = Vec::new();
    let mut dyn_geoms: Vec<Geom> = Vec::new();
    let (geoms, paints) = if layer.dynamic {
      let shapes: Vec<Value> = layer
        .shapes
        .as_ref()
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
      let local = self.frame - layer.time_offset;
      let (p, g) = walk_items(&shapes, Mat::identity(), 1.0, local);
      dyn_paints = p;
      dyn_geoms = g;
      (dyn_geoms.as_slice(), dyn_paints.as_slice())
    } else {
      (layer.geoms.as_slice(), layer.paints.as_slice())
    };

    for geom in geoms {
      if geom.verts.is_empty() {
        continue;
      }
      let first = self.vertices.len();
      for vertex in &geom.verts {
        let w = world.apply(*vertex);
        self.vertices.push([w[0] * sx, w[1] * sy]);
      }
      if let Some(image) = geom.image {
        let uv_first = self.uvs.len();
        self.uvs.extend_from_slice(&geom.uvs);
        self.batches.push(Batch {
          kind: 3,
          first,
          count: geom.verts.len(),
          color: [1.0, 1.0, 1.0, 1.0],
          alpha: layer_alpha,
          p0: [0.0, 0.0],
          p1: [0.0, 0.0],
          stops: Vec::new(),
          uv_first,
          group,
          matte,
          role,
          image: Some(image),
        });
        continue;
      }
      let Some(paint) = paints.get(geom.paint) else {
        continue;
      };
      let g0 = world.apply(paint.p0);
      let g1 = world.apply(paint.p1);
      self.batches.push(Batch {
        kind: paint.kind,
        first,
        count: geom.verts.len(),
        color: paint.color,
        alpha: paint.alpha * layer_alpha,
        p0: [g0[0] * sx, g0[1] * sy],
        p1: [g1[0] * sx, g1[1] * sy],
        stops: paint.stops.clone(),
        uv_first: 0,
        group,
        matte,
        role,
        image: None,
      });
    }
  }
}

fn render_frame(c: &Compiled, frame: f32) -> Vec<u8> {
  let mut r = Renderer::new(c, frame);
  let roots = c.roots.clone();
  r.render_list(&roots, 0, 0, 0);

  let mut words: Vec<u32> = Vec::with_capacity(
    3 + r.batches.len() * BATCH_WORDS + r.vertices.len() * 2 + r.uvs.len() * 2,
  );
  words.push(r.batches.len() as u32);
  words.push(r.vertices.len() as u32);
  words.push(r.uvs.len() as u32);
  for b in &r.batches {
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
    words.push(b.uv_first as u32);
    words.push(b.group);
    words.push(b.matte);
    words.push(b.role);
    words.push(b.image.map(|i| i as u32).unwrap_or(u32::MAX));
    // 58..63 保留
    for _ in 58..BATCH_WORDS {
      words.push(0);
    }
  }
  for v in &r.vertices {
    words.push(v[0].to_bits());
    words.push(v[1].to_bits());
  }
  for uv in &r.uvs {
    words.push(uv[0].to_bits());
    words.push(uv[1].to_bits());
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
        let plan = render_frame(c, frame);
        OUTPUT = plan;
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

#[no_mangle]
pub extern "C" fn l2s_image_count() -> u32 {
  unsafe {
    COMPILED
      .as_ref()
      .map(|c| c.images.len() as u32)
      .unwrap_or(0)
  }
}

#[no_mangle]
pub extern "C" fn l2s_image_src_ptr(index: u32) -> *const u8 {
  unsafe {
    COMPILED
      .as_ref()
      .and_then(|c| c.images.get(index as usize))
      .map(|i| i.src.as_ptr())
      .unwrap_or(std::ptr::null())
  }
}

#[no_mangle]
pub extern "C" fn l2s_image_src_len(index: u32) -> usize {
  unsafe {
    COMPILED
      .as_ref()
      .and_then(|c| c.images.get(index as usize))
      .map(|i| i.src.len())
      .unwrap_or(0)
  }
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
  use super::*;

  fn base_layer(shapes: Value) -> Value {
    serde_json::json!({
      "ddd": 0, "ind": 1, "ty": 4, "nm": "s", "sr": 1,
      "ks": {
        "o": {"a": 0, "k": 100},
        "r": {"a": 0, "k": 0},
        "p": {"a": 0, "k": [200, 200, 0]},
        "a": {"a": 0, "k": [0, 0, 0]},
        "s": {"a": 0, "k": [100, 100, 100]}
      },
      "ao": 0, "shapes": shapes, "ip": 0, "op": 60, "st": 0, "bm": 0
    })
  }

  fn rect_group(fill: Value) -> Value {
    let items = vec![
      serde_json::json!({"ty": "rc", "p": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [200, 120]}, "r": {"a": 0, "k": 0}}),
      fill,
      serde_json::json!({"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}}),
    ];
    serde_json::json!({"ty": "gr", "it": items, "nm": "g"})
  }

  fn read_words(bytes: &[u8]) -> Vec<u32> {
    bytes
      .chunks_exact(4)
      .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
      .collect()
  }

  fn vertex_base(words: &[u32]) -> usize {
    3 + words[0] as usize * BATCH_WORDS
  }

  fn read_vertex(words: &[u32], base: usize, index: usize) -> [f32; 2] {
    let at = base + index * 2;
    [f32::from_bits(words[at]), f32::from_bits(words[at + 1])]
  }

  #[test]
  fn parses_and_renders_simple_rect() {
    let doc = serde_json::json!({
      "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
      "layers": [base_layer(serde_json::json!([rect_group(
        serde_json::json!({"ty": "fl", "c": {"a": 0, "k": [0.2, 0.6, 1, 1]}, "o": {"a": 0, "k": 100}})
      )]))]
    });
    let compiled = compile_doc(&doc, 512.0, 512.0).expect("compile");
    let bytes = render_frame(&compiled, 0.0);
    let words = read_words(&bytes);
    assert!(words.len() >= 3);
    assert!(words[0] >= 1, "至少应产出一个绘制批次");
    assert!(words[1] >= 6, "顶点数量应大于 0");
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
    let doc = serde_json::json!({
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
    let compiled = compile_doc(&doc, 400.0, 400.0).expect("compile");
    let bytes = render_frame(&compiled, 0.0);
    let words = read_words(&bytes);
    let batch_count = words[0] as usize;
    let base = vertex_base(&words);
    let mut area = 0.0f32;
    for b in 0..batch_count {
      let header = 3 + b * BATCH_WORDS;
      let vertex_start = words[header + 1] as usize;
      let vertex_count = words[header + 2] as usize;
      for tri in (0..vertex_count).step_by(3) {
        let a = read_vertex(&words, base, vertex_start + tri);
        let p1 = read_vertex(&words, base, vertex_start + tri + 1);
        let p2 = read_vertex(&words, base, vertex_start + tri + 2);
        area += (((p1[0] - a[0]) * (p2[1] - a[1]) - (p2[0] - a[0]) * (p1[1] - a[1])).abs()) * 0.5;
      }
    }
    let expected = std::f32::consts::PI * 100.0 * 100.0;
    let ratio = area / expected;
    assert!(ratio > 0.97 && ratio < 1.03, "椭圆面积比 {ratio}");
  }

  #[test]
  fn precomp_layers_are_expanded_with_transform() {
    let doc = serde_json::json!({
      "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
      "assets": [{
        "id": "comp1",
        "layers": [{
          "ddd": 0, "ind": 1, "ty": 4, "nm": "inner", "sr": 1,
          "ks": {
            "o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0},
            "p": {"a": 0, "k": [0, 0, 0]}, "a": {"a": 0, "k": [0, 0, 0]},
            "s": {"a": 0, "k": [100, 100, 100]}
          },
          "shapes": [rect_group(
            serde_json::json!({"ty": "fl", "c": {"a": 0, "k": [1, 0, 0, 1]}, "o": {"a": 0, "k": 100}})
          )],
          "ip": 0, "op": 60
        }]
      }],
      "layers": [{
        "ddd": 0, "ind": 1, "ty": 0, "refId": "comp1", "nm": "pre",
        "ks": {
          "o": {"a": 0, "k": 100}, "r": {"a": 0, "k": 0},
          "p": {"a": 0, "k": [200, 200, 0]}, "a": {"a": 0, "k": [0, 0, 0]},
          "s": {"a": 0, "k": [100, 100, 100]}
        },
        "ip": 0, "op": 60, "st": 0
      }]
    });
    let compiled = compile_doc(&doc, 400.0, 400.0).expect("compile");
    let bytes = render_frame(&compiled, 0.0);
    let words = read_words(&bytes);
    assert!(words[0] >= 1, "预合成内部图层应产生批次");
    // 预合成位移 200 → 内容应落在画布中心附近
    let base = vertex_base(&words);
    let count = words[1] as usize;
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    for i in 0..count {
      let v = read_vertex(&words, base, i);
      min_x = min_x.min(v[0]);
      max_x = max_x.max(v[0]);
    }
    assert!(min_x < 160.0 && max_x > 240.0, "预合成变换未生效：{min_x}..{max_x}");
  }

  #[test]
  fn gradient_stroke_produces_gradient_batch() {
    let doc = serde_json::json!({
      "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
      "layers": [base_layer(serde_json::json!([{
        "ty": "gr",
        "it": [
          {"ty": "rc", "p": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [200, 120]}, "r": {"a": 0, "k": 0}},
          {"ty": "gs", "s": {"a": 0, "k": [0, 0]}, "e": {"a": 0, "k": [200, 0]}, "t": 1, "w": {"a": 0, "k": 8},
           "g": {"p": 2, "k": {"a": 0, "k": [0, 1, 0, 0, 1, 1, 0, 0, 1, 1]}}, "o": {"a": 0, "k": 100}},
          {"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}}
        ]
      }]))]
    });
    let compiled = compile_doc(&doc, 400.0, 400.0).expect("compile");
    let bytes = render_frame(&compiled, 0.0);
    let words = read_words(&bytes);
    let batch_count = words[0] as usize;
    let mut has_gradient = false;
    for b in 0..batch_count {
      let header = 3 + b * BATCH_WORDS;
      if words[header] == 1 {
        has_gradient = true;
      }
    }
    assert!(has_gradient, "渐变描边应输出 kind=1 批次");
  }

  #[test]
  fn trim_path_shortens_stroke() {
    let make = |end: f32| {
      serde_json::json!({
        "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
        "layers": [base_layer(serde_json::json!([{
          "ty": "gr",
          "it": [
            {"ty": "rc", "p": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [200, 120]}, "r": {"a": 0, "k": 0}},
            {"ty": "tm", "s": {"a": 0, "k": 0}, "e": {"a": 0, "k": end}, "o": {"a": 0, "k": 0}, "m": 1},
            {"ty": "st", "c": {"a": 0, "k": [1, 0, 0, 1]}, "o": {"a": 0, "k": 100}, "w": {"a": 0, "k": 6}},
            {"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}}
          ]
        }]))]
      })
    };
    let full = render_frame(&compile_doc(&make(100.0), 400.0, 400.0).unwrap(), 0.0);
    let half = render_frame(&compile_doc(&make(50.0), 400.0, 400.0).unwrap(), 0.0);
    let full_words = read_words(&full);
    let half_words = read_words(&half);
    assert!(
      half_words[1] < full_words[1],
      "trim 50% 的顶点数应少于完整路径：{} vs {}",
      half_words[1],
      full_words[1]
    );
  }

  #[test]
  fn track_matte_marks_source_and_masked_batches() {
    let masked = {
      let mut layer = base_layer(serde_json::json!([rect_group(
        serde_json::json!({"ty": "fl", "c": {"a": 0, "k": [0, 1, 0, 1]}, "o": {"a": 0, "k": 100}})
      )]));
      layer["ind"] = serde_json::json!(2);
      layer["tt"] = serde_json::json!(1);
      layer
    };
    let source = {
      let mut layer = base_layer(serde_json::json!([rect_group(
        serde_json::json!({"ty": "fl", "c": {"a": 0, "k": [1, 0, 0, 1]}, "o": {"a": 0, "k": 100}})
      )]));
      layer["ind"] = serde_json::json!(1);
      layer["td"] = serde_json::json!(1);
      layer
    };
    let doc = serde_json::json!({
      "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
      "layers": [source, masked]
    });
    let compiled = compile_doc(&doc, 400.0, 400.0).expect("compile");
    let bytes = render_frame(&compiled, 0.0);
    let words = read_words(&bytes);
    let batch_count = words[0] as usize;
    let mut source_batches = 0;
    let mut masked_batches = 0;
    let mut group = 0;
    for b in 0..batch_count {
      let header = 3 + b * BATCH_WORDS;
      let role = words[header + 56];
      if role == 1 {
        source_batches += 1;
        group = words[header + 54];
      }
      if role == 2 {
        masked_batches += 1;
        assert_eq!(words[header + 54], group, "遮罩两端应属于同一分组");
        assert_eq!(words[header + 55], 1, "alpha 遮罩模式");
      }
    }
    assert!(source_batches > 0 && masked_batches > 0, "缺少遮罩批次");
  }

  #[test]
  fn three_d_rotation_narrows_projection() {
    let mut layer = base_layer(serde_json::json!([rect_group(
      serde_json::json!({"ty": "fl", "c": {"a": 0, "k": [1, 0, 0, 1]}, "o": {"a": 0, "k": 100}})
    )]));
    layer["ddd"] = serde_json::json!(1);
    layer["pe"] = serde_json::json!(800);
    layer["ks"]["ry"] = serde_json::json!({"a": 0, "k": 60});
    let doc = serde_json::json!({
      "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
      "layers": [layer]
    });
    let compiled = compile_doc(&doc, 400.0, 400.0).expect("compile");
    let bytes = render_frame(&compiled, 0.0);
    let words = read_words(&bytes);
    let base = vertex_base(&words);
    let count = words[1] as usize;
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    for i in 0..count {
      let v = read_vertex(&words, base, i);
      min_x = min_x.min(v[0]);
      max_x = max_x.max(v[0]);
    }
    assert!(
      max_x - min_x < 170.0,
      "绕 Y 轴旋转 60 度后宽度应变窄，实际 {}",
      max_x - min_x
    );
  }

  #[test]
  fn unsupported_features_fail_explicitly() {
    let mut doc = serde_json::json!({
      "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
      "layers": [base_layer(serde_json::json!([rect_group(
        serde_json::json!({"ty": "fl", "c": {"a": 0, "k": [1, 0, 0, 1]}, "o": {"a": 0, "k": 100}})
      )]))]
    });
    doc["layers"][0]["ty"] = serde_json::json!(5);
    let err = match compile_doc(&doc, 400.0, 400.0) {
      Err(message) => message,
      Ok(_) => panic!("文本图层应当编译失败"),
    };
    assert!(err.contains("文本图层"), "实际错误：{err}");
  }
}
