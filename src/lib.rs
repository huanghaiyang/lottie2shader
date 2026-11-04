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
//! - 合并路径（mm，仅 Merge 模式：把组内子路径合并后用组的填充规则整体三角化）；
//! - 轨道遮罩（tt=1/2，alpha 与反相 alpha）；亮度遮罩不支持；
//! - 3D 图层的 rx/ry/rz 旋转与透视投影（不含 3D 相机）；
//! - 图层与组的变换、不透明度关键帧、父子关系、预合成嵌套。
//!
//! 明确不支持的特性会直接返回错误，由前端注册中心回退到 canvas，绝不静默画错：
//! 文本图层（需要字体轮廓）、表达式（需要 JS 引擎）、图层蒙版、图层特效、
//! 3D 相机、亮度轨道遮罩、星形/重复器等形状修饰器，以及合并路径的布尔模式
//! （mm=2/3/4/5，Add/Subtract/Intersect/Exclude）。

use std::collections::{HashMap, HashSet};

use lyon::algorithms::measure::{PathMeasurements, SampleType};
use lyon::math::Point;
use lyon::path::{Path, PathEvent};
use lyon::tessellation::geometry_builder::{BuffersBuilder, VertexBuffers};
use lyon::tessellation::{
  FillOptions, FillRule, FillTessellator, FillVertex, LineCap, LineJoin, StrokeOptions,
  StrokeTessellator, StrokeVertex,
};
use serde_json::Value;

/// 单个批次最多承载的渐变色标数量，与前端着色器 uniform 数组长度保持一致
const MAX_STOPS: usize = 16;
/// 批次头固定占用字（u32）数 = 13 + MAX_STOPS*5 + 9，必须与前端 BATCH_WORDS 一致
const BATCH_WORDS: usize = 13 + MAX_STOPS * 5 + 11;
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
/// 蒙版覆盖几何：只输出 alpha（与图层内容一起合成）
const KIND_MASK: u32 = 5;
/// 全屏覆盖几何：给反向蒙版的「加」模式铺底
const KIND_FULL: u32 = 6;
/// 无 passId 的批次（直接画到父目标）
const PASS_NONE: u32 = u32::MAX;

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
  // Lottie 里不少数值位会用布尔表示（en、inv、sz、mm 等），按 1/0 解析
  match v {
    Some(Value::Bool(flag)) => {
      if *flag {
        1.0
      } else {
        0.0
      }
    }
    Some(other) => other
      .as_f64()
      .map(|x| x as f32)
      .unwrap_or(def),
    None => def,
  }
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

/// 从缓动控制点里读取一个分量：变换类关键帧写成标量 `{"x":0.358,"y":0.965}`，
/// 形状 / 蒙版类关键帧写成数组 `{"x":[...],"y":[...]}`，两种都要支持。
fn easing_pair(v: Option<&Value>) -> (f32, f32) {
  fn component(e: Option<&Value>, key: &str) -> f32 {
    match e.and_then(|e| e.get(key)) {
      Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0) as f32,
      Some(Value::Array(arr)) => arr
        .first()
        .and_then(|x| x.as_f64())
        .unwrap_or(0.0) as f32,
      _ => 0.0,
    }
  }
  let x = component(v, "x");
  let y = component(v, "y");
  // 部分导出把控制点写成百分比（0~100），统一归一化
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
  /// 1 alpha、2 反相 alpha、3 亮度、4 反相亮度
  matte_mode: u32,
  /// 被遮罩图层对应的遮罩源图层下标（role == 2 时有效）
  matte_src: Option<usize>,
  /// 图层混合模式（lottie 的 bm 原值，0 表示正常混合）
  blend: u32,
  /// 图层高斯模糊（ef 29），可能为动画属性
  blur: Option<BlurFx>,
  /// 图层蒙版（masksProperties）
  masks: Vec<MaskSpec>,
}

/// 图层高斯模糊参数（数值保持原始属性，渲染时再按帧求值）
struct BlurFx {
  blurriness: Value,
  dimensions: Value,
  edge: Value,
}

/// 一条图层蒙版
#[derive(Clone)]
struct MaskSpec {
  /// 遮罩路径属性（pt）
  path: Value,
  /// 0 加 / 1 减 / 2 交 / 3 忽略
  mode: u32,
  /// 反向蒙版
  inv: bool,
  /// 不透明度（百分比属性）
  opacity: Value,
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

/// mm=1（Merge）：把组内子路径合并成一条路径。
/// Merge 不做布尔运算，只把各子轮廓归入同一条路径，由后续填充规则（nonzero/evenodd）
/// 统一判定内部区域，因此空洞、重叠等由轮廓绕向自然表达。
fn merge_paths(paths: &mut Vec<Path>) {
  if paths.len() <= 1 {
    return;
  }
  let mut builder = Path::builder();
  for path in paths.iter() {
    for event in path.iter() {
      match event {
        PathEvent::Begin { at } => {
          builder.begin(at);
        }
        PathEvent::Line { to, .. } => {
          builder.line_to(to);
        }
        PathEvent::Quadratic { ctrl, to, .. } => {
          builder.quadratic_bezier_to(ctrl, to);
        }
        PathEvent::Cubic { ctrl1, ctrl2, to, .. } => {
          builder.cubic_bezier_to(ctrl1, ctrl2, to);
        }
        PathEvent::End { close, .. } => {
          builder.end(close);
        }
      }
    }
  }
  let merged = builder.build();
  paths.clear();
  paths.push(merged);
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
  // 每个顶点的入切线：顶点 i 的入切线对应上一段（圆弧）结束处的控制柄
  let ins = [
    [-k, 0.0],
    [0.0, 0.0],
    [0.0, -k],
    [0.0, 0.0],
    [k, 0.0],
    [0.0, 0.0],
    [0.0, k],
    [0.0, 0.0],
  ];
  bezier_path(&verts, &ins, &outs, true, mat)
}

// ---------------------------------------------------------------------------
// 多边形 / 星形（polystar）几何
// ---------------------------------------------------------------------------

fn path_from_star(item: &Value, mat: &Mat, frame: f32) -> Option<Path> {
  let star = as_f32(item.get("sy"), 1.0) <= 1.5;
  let points = prop_f32(item.get("pt"), frame, 5.0).floor().max(1.0) as usize;
  let pos = prop_vec(item.get("p"), frame, 2, &[0.0, 0.0]);
  let rot = prop_f32(item.get("r"), frame, 0.0).to_radians();
  let dir = if as_f32(item.get("d"), 1.0) >= 3.0 { -1.0 } else { 1.0 };
  let outer = prop_f32(item.get("or"), frame, 0.0);
  let inner = prop_f32(item.get("ir"), frame, 0.0);
  let os = prop_f32(item.get("os"), frame, 0.0) / 100.0;
  let round_inner = prop_f32(item.get("is"), frame, 0.0) / 100.0;
  let count = if star { points * 2 } else { points };
  let angle = std::f32::consts::PI * 2.0 / count as f32;
  let mut start = -std::f32::consts::PI * 0.5 + rot;
  let mut verts: Vec<[f32; 2]> = Vec::with_capacity(count);
  let mut ins: Vec<[f32; 2]> = Vec::with_capacity(count);
  let mut outs: Vec<[f32; 2]> = Vec::with_capacity(count);
  for i in 0..count {
    let long = star && i % 2 == 0;
    let rad = if long || !star { outer } else { inner };
    let roundness = if long || !star { os } else { round_inner };
    let perim = if long || !star {
      2.0 * std::f32::consts::PI * outer / (count * 2) as f32
    } else {
      2.0 * std::f32::consts::PI * inner / (count * 2) as f32
    };
    let x = rad * start.cos();
    let y = rad * start.sin();
    let len = (x * x + y * y).sqrt();
    let (ox, oy) = if len < 1.0e-6 {
      (0.0, 0.0)
    } else {
      (y / len, -x / len)
    };
    verts.push([x + pos[0], y + pos[1]]);
    ins.push([ox * perim * roundness * dir, oy * perim * roundness * dir]);
    outs.push([-ox * perim * roundness * dir, -oy * perim * roundness * dir]);
    start += angle * dir;
  }
  bezier_path(&verts, &ins, &outs, true, mat)
}


/// lottie 的圆角常量（四分之一圆的贝塞尔控制点比例）
const ROUND_CORNER: f32 = 0.5519;

/// 一个轮廓顶点及其入/出控制点（绝对坐标）
struct CornerVertex {
  p: [f32; 2],
  i: [f32; 2],
  o: [f32; 2],
}

#[inline]
fn near_point(a: [f32; 2], b: [f32; 2]) -> bool {
  (a[0] - b[0]).abs() < 0.01 && (a[1] - b[1]).abs() < 0.01
}

/// 把 lyon 路径拆成「顶点 + 入/出控制点」的轮廓集合，便于做圆角处理
fn path_contours(path: &Path) -> Vec<(Vec<CornerVertex>, bool)> {
  let mut out: Vec<(Vec<CornerVertex>, bool)> = Vec::new();
  let mut current: Vec<CornerVertex> = Vec::new();
  for event in path.iter() {
    match event {
      PathEvent::Begin { at } => {
        current = vec![CornerVertex {
          p: [at.x, at.y],
          i: [at.x, at.y],
          o: [at.x, at.y],
        }];
      }
      PathEvent::Line { from, to } => {
        if let Some(v) = current.last_mut() {
          v.o = [from.x, from.y];
        }
        current.push(CornerVertex {
          p: [to.x, to.y],
          i: [to.x, to.y],
          o: [to.x, to.y],
        });
      }
      PathEvent::Quadratic { from, ctrl, to } => {
        if let Some(v) = current.last_mut() {
          v.o = [from.x + (ctrl.x - from.x) * 2.0 / 3.0, from.y + (ctrl.y - from.y) * 2.0 / 3.0];
        }
        current.push(CornerVertex {
          p: [to.x, to.y],
          i: [to.x + (ctrl.x - to.x) * 2.0 / 3.0, to.y + (ctrl.y - to.y) * 2.0 / 3.0],
          o: [to.x, to.y],
        });
      }
      PathEvent::Cubic {
        ctrl1,
        ctrl2,
        to,
        ..
      } => {
        if let Some(v) = current.last_mut() {
          v.o = [ctrl1.x, ctrl1.y];
        }
        current.push(CornerVertex {
          p: [to.x, to.y],
          i: [ctrl2.x, ctrl2.y],
          o: [to.x, to.y],
        });
      }
      PathEvent::End { close, .. } => {
        let closed = close;
        if closed && current.len() > 1 {
          let first = current[0].p;
          let last = current.last().unwrap();
          if near_point(first, last.p) {
            // 闭合路径的末顶点与首顶点重合：把末顶点的入切向并入首顶点
            current[0].i = last.i;
            current.pop();
          }
        }
        if current.len() >= 2 {
          out.push((std::mem::take(&mut current), closed));
        } else {
          current.clear();
        }
      }
    }
  }
  out
}

/// 圆角修饰器（rd）：对所有「无控制柄」的尖角做倒角。
/// 与 lottie 一致：单边切掉长度取 min(半径, 相邻边长/2)，圆弧用 0.5519 常量近似。
fn round_corners(path: &Path, radius: f32) -> Option<Path> {
  if radius <= 0.01 {
    return None;
  }
  let contours = path_contours(path);
  let mut builder = Path::builder();
  let mut changed = false;
  for (contour, closed) in contours {
    let n = contour.len();
    if n < 2 {
      continue;
    }
    let mut verts: Vec<[f32; 2]> = Vec::with_capacity(n * 2);
    let mut ins: Vec<[f32; 2]> = Vec::with_capacity(n * 2);
    let mut outs: Vec<[f32; 2]> = Vec::with_capacity(n * 2);
    for idx in 0..n {
      let v = &contour[idx];
      // 开放路径的首末顶点没有完整的两条边，不能倒角
      if (!closed && (idx == 0 || idx == n - 1)) || !near_point(v.i, v.p) || !near_point(v.o, v.p) {
        verts.push(v.p);
        ins.push([v.i[0] - v.p[0], v.i[1] - v.p[1]]);
        outs.push([v.o[0] - v.p[0], v.o[1] - v.p[1]]);
        continue;
      }
      let prev = &contour[(idx + n - 1) % n];
      let next = &contour[(idx + 1) % n];
      let d_prev = ((v.p[0] - prev.p[0]).powi(2) + (v.p[1] - prev.p[1]).powi(2)).sqrt();
      let d_next = ((next.p[0] - v.p[0]).powi(2) + (next.p[1] - v.p[1]).powi(2)).sqrt();
      let cut_prev = radius.min(d_prev / 2.0);
      let cut_next = radius.min(d_next / 2.0);
      if cut_prev < 0.01 || cut_next < 0.01 {
        verts.push(v.p);
        ins.push([0.0, 0.0]);
        outs.push([0.0, 0.0]);
        continue;
      }
      let p1 = [
        v.p[0] + (prev.p[0] - v.p[0]) / d_prev * cut_prev,
        v.p[1] + (prev.p[1] - v.p[1]) / d_prev * cut_prev,
      ];
      let p2 = [
        v.p[0] + (next.p[0] - v.p[0]) / d_next * cut_next,
        v.p[1] + (next.p[1] - v.p[1]) / d_next * cut_next,
      ];
      // p1 出切向与 p2 入切向都指向原尖角，形成圆弧
      verts.push(p1);
      ins.push([0.0, 0.0]);
      outs.push([(v.p[0] - p1[0]) * ROUND_CORNER, (v.p[1] - p1[1]) * ROUND_CORNER]);
      verts.push(p2);
      ins.push([(v.p[0] - p2[0]) * ROUND_CORNER, (v.p[1] - p2[1]) * ROUND_CORNER]);
      outs.push([0.0, 0.0]);
      changed = true;
    }
    let m = verts.len();
    if m < 2 {
      continue;
    }
    if m == 2 {
      builder.begin(to_point(&Mat::identity(), verts[0]));
      builder.line_to(to_point(&Mat::identity(), verts[1]));
      builder.end(closed);
      continue;
    }
    builder.begin(to_point(&Mat::identity(), verts[0]));
    let segments = if closed { m } else { m - 1 };
    for i in 0..segments {
      let j = (i + 1) % m;
      let c1 = [verts[i][0] + outs[i][0], verts[i][1] + outs[i][1]];
      let c2 = [verts[j][0] + ins[j][0], verts[j][1] + ins[j][1]];
      let straight = (c1[0] - verts[i][0]).abs() < 0.01
        && (c1[1] - verts[i][1]).abs() < 0.01
        && (c2[0] - verts[j][0]).abs() < 0.01
        && (c2[1] - verts[j][1]).abs() < 0.01;
      if straight {
        builder.line_to(to_point(&Mat::identity(), verts[j]));
      } else {
        builder.cubic_bezier_to(
          to_point(&Mat::identity(), c1),
          to_point(&Mat::identity(), c2),
          to_point(&Mat::identity(), verts[j]),
        );
      }
    }
    builder.end(closed);
  }
  changed.then(|| builder.build())
}

/// 重复器（rp）第 t 个副本的变换：平移 p*t、绕锚点旋转 -r*t、按 s^t 缩放
fn repeater_matrix(item: &Value, frame: f32, t: f32) -> Mat {
  let tr = item.get("tr");
  let p = prop_vec(tr.and_then(|t| t.get("p")), frame, 2, &[0.0, 0.0]);
  let a = prop_vec(tr.and_then(|t| t.get("a")), frame, 2, &[0.0, 0.0]);
  let s = prop_vec(tr.and_then(|t| t.get("s")), frame, 2, &[100.0, 100.0]);
  let r = prop_f32(tr.and_then(|t| t.get("r")), frame, 0.0).to_radians();
  let sx = s[0] / 100.0;
  let sy = s[1] / 100.0;
  let whole = t.floor();
  let frac = t - whole;
  // lottie 逐副本累乘缩放；小数偏移用线性插值补齐
  let fx = sx.powf(whole) * (sx + (1.0 - sx) * (1.0 - frac));
  let fy = sy.powf(whole) * (sy + (1.0 - sy) * (1.0 - frac));
  let anchor = Mat::translate(a[0], a[1]);
  let back = Mat::translate(-a[0], -a[1]);
  let rot = anchor.mul(&Mat::rotate(-r * t)).mul(&back);
  let scale = anchor.mul(&Mat::scale(fx, fy)).mul(&back);
  let offset = Mat::translate(p[0] * t, p[1] * t);
  rot.mul(&scale).mul(&offset)
}

/// 重复器每个副本的不透明度：so → eo 线性过渡
fn repeater_opacity(item: &Value, frame: f32, index: f32, copies: f32) -> f32 {
  let tr = item.get("tr");
  let so = prop_f32(tr.and_then(|t| t.get("so")), frame, 100.0) / 100.0;
  let eo = prop_f32(tr.and_then(|t| t.get("eo")), frame, 100.0) / 100.0;
  if copies <= 1.0 {
    return so;
  }
  so + (eo - so) * (index / (copies - 1.0))
}

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

/// 按位置 t 从色标序列中取色（用于超出着色器上限时的重采样）
fn sample_gradient(stops: &[[f32; 5]], t: f32) -> [f32; 5] {
  if stops.is_empty() {
    return [0.0, 0.0, 0.0, 0.0, 1.0];
  }
  if t <= stops[0][0] {
    return stops[0];
  }
  for i in 1..stops.len() {
    if t < stops[i][0] {
      let span = (stops[i][0] - stops[i - 1][0]).max(1e-5);
      let f = ((t - stops[i - 1][0]) / span).clamp(0.0, 1.0);
      let mut out = stops[i - 1];
      for c in 1..5 {
        out[c] = stops[i - 1][c] + (stops[i][c] - stops[i - 1][c]) * f;
      }
      out[0] = t;
      return out;
    }
  }
  let mut last = *stops.last().unwrap();
  last[0] = t;
  last
}

/// 渐变画笔（gf 填充 / gs 描边共用）
fn gradient_paint(item: &Value, alpha: f32, frame: f32) -> Paint {
  let s = prop_vec(item.get("s"), frame, 2, &[0.0, 0.0]);
  let e = prop_vec(item.get("e"), frame, 2, &[1.0, 0.0]);
  // Lottie：t=1 线性，t=2 径向。`r` 在 gf 上是填充规则（1 = nonzero），不是径向标志
  let is_radial = as_f32(item.get("t"), 0.0) == 2.0;
  let mut stops: Vec<[f32; 5]> = Vec::new();
  if let Some(g) = item.get("g") {
    let count = as_f32(g.get("p"), 0.0) as usize;
    let values = prop_f32_array(g.get("k"), frame);
    // 色标可能是 4 分量（rgb）或 5 分量（rgba），按总长度判断
    if !values.is_empty() {
      let stride = if count > 0 && values.len() == count * 5 { 5 } else { 4 };
      // 导出器有时会在 k 里追加第二组渐变（长度翻倍），必须按 p 截断
      let total = if count > 0 { count } else { values.len() / stride };
      let mut all: Vec<[f32; 5]> = Vec::new();
      for s in 0..total {
        let i = s * stride;
        if i + stride > values.len() {
          break;
        }
        let offset = values[i];
        let r = values[i + 1];
        let g = values[i + 2];
        let b = values[i + 3];
        let a = if stride == 5 { values[i + 4] } else { 1.0 };
        all.push([offset, r, g, b, a]);
      }
      if all.len() > MAX_STOPS {
        // 着色器只接受 MAX_STOPS 个色标，按位置均匀重采样，避免直接截断丢掉尾部颜色
        for k in 0..MAX_STOPS {
          let t = k as f32 / (MAX_STOPS - 1) as f32;
          stops.push(sample_gradient(&all, t));
        }
      } else {
        stops = all;
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
) -> (Vec<Paint>, Vec<Geom>, Vec<Path>) {
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

  for (index, item) in items.iter().enumerate() {
    let ty = item.get("ty").and_then(|t| t.as_str()).unwrap_or("");
    match ty {
      "rp" => {
        // 重复器：复制它前面的全部形状与样式，每个副本套一层变换与不透明度
        let copies_f = prop_f32(item.get("c"), frame, 1.0).max(0.0).ceil();
        let copies = copies_f as usize;
        if index > 0 && copies > 0 {
          let prefix: Vec<Value> = items[..index].to_vec();
          let offset = prop_f32(item.get("o"), frame, 0.0);
          let reversed = as_f32(item.get("m"), 1.0) >= 2.0;
          // 原先按副本 1 处理的内容要整体丢弃，改为按副本重放
          paints.clear();
          geoms.clear();
          paths.clear();
          for step in 0..copies {
            let copy = if reversed { copies - 1 - step } else { step } as f32;
            let copy_mat = mat.mul(&repeater_matrix(item, frame, copy + offset));
            let copy_alpha = group_alpha * repeater_opacity(item, frame, copy, copies_f);
            let (child_paints, mut child_geoms, _) = walk_items(&prefix, copy_mat, copy_alpha, frame);
            let paint_offset = paints.len();
            for geom in &mut child_geoms {
              if geom.image.is_none() {
                geom.paint += paint_offset;
              }
            }
            paints.extend(child_paints);
            geoms.extend(child_geoms);
          }
        }
      }
      "rd" => {
        let radius = prop_f32(item.get("r"), frame, 0.0).max(0.0) * mat.scale_factor();
        paths = std::mem::take(&mut paths)
          .into_iter()
          .map(|p| round_corners(&p, radius).unwrap_or(p))
          .collect();
      }
      "gr" => {
        if let Some(child) = item.get("it").and_then(|i| i.as_array()) {
          let (child_paints, mut child_geoms, child_paths) = walk_items(child, mat, group_alpha, frame);
          if child_geoms.is_empty() {
            // 子组没有任何画笔时，其路径并入本组，由本组的填充 / 描边统一绘制
            // （AE 导出常见：路径分组 + 组级 Gradient Fill）
            paths.extend(child_paths);
          } else {
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
      "sr" => {
        if let Some(p) = path_from_star(item, &mat, frame) {
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
      "mm" => {
        // 合并模式在 check_shapes 已校验，这里只处理 mm=1（Merge）
        merge_paths(&mut paths);
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
  (paints, geoms, paths)
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
      "mm" => {
        // 仅实现 Merge（mm=1）；Add/Subtract/Intersect/Exclude 需要布尔运算
        if as_f32(item.get("mm"), 1.0) >= 1.5 {
          return Err("不支持的合并路径模式（仅支持 Merge，请回退 canvas 渲染）".to_string());
        }
      }
      "rd" => {
        // 圆角修饰器：半径可为 0、可动画；渲染时按帧在路径上倒角
      }
      "rp" => {
        // 重复器：按副本数复制前方的形状与样式
        if prop_f32(item.get("c"), 0.0, 1.0) > 1000.0 {
          return Err("重复器副本数过多".to_string());
        }
      }
      "sh" | "el" | "rc" | "sr" | "fl" | "st" | "gf" | "gs" | "tr" | "tm" | "no" => {}
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

/// 图层特效里处于启用状态的项（缺省视为启用，en=0 视为关闭）
fn enabled_effects(layer: &Value) -> Vec<&Value> {
  layer
    .get("ef")
    .and_then(|e| e.as_array())
    .map(|arr| arr.iter().filter(|ef| as_f32(ef.get("en"), 1.0) > 0.5).collect())
    .unwrap_or_default()
}

/// 解析高斯模糊（ef ty=29）：lottie 的 effectElements 顺序是
/// [Blurriness, Blur Dimensions, Edge Mode]，直接按位置取值。
fn layer_blur(layer: &Value) -> Option<BlurFx> {
  let effects = enabled_effects(layer);
  let effect = effects.iter().find(|ef| as_f32(ef.get("ty"), 0.0) == 29.0)?;
  let params = effect.get("ef")?.as_array()?;
  // 参数缺省时按 lottie 默认值补齐，避免「有模糊效果却未生效」的静默错误
  let value = |i: usize, def: f32| -> Value {
    params
      .get(i)
      .and_then(|p| p.get("v"))
      .cloned()
      .unwrap_or(Value::from(def))
  };
  Some(BlurFx {
    blurriness: value(0, 0.0),
    dimensions: value(1, 1.0),
    edge: value(2, 0.0),
  })
}

/// 解析图层蒙版；mode 映射为 0 加 / 1 减 / 2 交 / 3 忽略
fn layer_masks(layer: &Value) -> Vec<MaskSpec> {
  let Some(masks) = layer.get("masksProperties").and_then(|m| m.as_array()) else {
    return Vec::new();
  };
  masks
    .iter()
    .filter_map(|mask| {
      let mode = match mask.get("mode").and_then(|m| m.as_str()).unwrap_or("a") {
        "s" => 1,
        "i" => 2,
        "n" => 3,
        _ => 0,
      };
      Some(MaskSpec {
        path: mask.get("pt")?.clone(),
        mode,
        inv: as_f32(mask.get("inv"), 0.0) > 0.5,
        opacity: mask.get("o").cloned().unwrap_or(Value::Null),
      })
    })
    .collect()
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
      for mask in masks {
        let mode = mask.get("mode").and_then(|m| m.as_str()).unwrap_or("a");
        if !matches!(mode, "a" | "s" | "i" | "n") {
          return Err(format!("不支持的图层蒙版模式：{mode}（请回退 canvas 渲染）"));
        }
        // lottie 用 feMorphology 实现蒙版扩张/收缩，GPU 侧没有等价实现
        if prop_f32(mask.get("x"), 0.0, 0.0).abs() > 1.0e-6 {
          return Err("图层蒙版的扩张/收缩（x）暂不支持（请回退 canvas 渲染）".to_string());
        }
        if mask.get("pt").is_none() {
          return Err("图层蒙版缺少路径属性 pt".to_string());
        }
      }
    }
    let effects = enabled_effects(layer);
    if !effects.is_empty() {
      if effects.len() > 1 {
        return Err("多个图层特效叠加暂不支持（请回退 canvas 渲染）".to_string());
      }
      let ty_effect = as_f32(effects[0].get("ty"), 0.0) as i64;
      if ty_effect != 29 {
        return Err(format!("不支持的图层特效 ty={ty_effect}（仅支持高斯模糊）"));
      }
      let params = effects[0]
        .get("ef")
        .and_then(|p| p.as_array())
        .ok_or_else(|| "高斯模糊缺少参数".to_string())?;
      if params.len() < 2 || params.first().and_then(|p| p.get("v")).is_none() {
        return Err("高斯模糊参数不完整".to_string());
      }
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
    matte_src: None,
    blend: 0,
    blur: None,
    masks: Vec::new(),
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
    let blend = layer.get("bm").and_then(|b| b.as_f64()).unwrap_or(0.0) as u32;
    // 预合成图层的内容时间轴从 st 起算；普通图层的属性关键帧按当前帧直接求值
    let st = as_f32(layer.get("st"), 0.0);

    let slot = out.len();
    out.push(placeholder(container, time_offset, ip, op, ks.clone(), ddd, pe));
    entry_of_raw[idx] = Some(slot);
    entries.push(slot);

    let mut entry = placeholder(container, time_offset, ip, op, ks, ddd, pe);
    entry.blend = blend;
    entry.blur = layer_blur(layer);
    entry.masks = layer_masks(layer);
    match ty {
      4 => {
        let shapes = layer
          .get("shapes")
          .and_then(|s| s.as_array())
          .cloned()
          .unwrap_or_default();
        let dynamic = shapes.iter().any(has_animation);
        let (paints, geoms, _) = walk_items(&shapes, Mat::identity(), 1.0, ip);
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
        // 预合成的内容时间轴从本层的 st 起算
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

  // 轨道遮罩配对：
  // - `tp` 显式给出遮罩源的图层 ind（同一 scope 内查找）；
  // - 否则沿用 lottie 的隐式规则：遮罩源是 ind - 1（图层列表中紧邻其上的那层）。
  // 注意配对依据是 ind 而不是数组下标，且源图层不必带 td（td 只标记“这是遮罩源”）。
  // 同一个遮罩源可以被多个被遮罩图层复用，因此不能把分组 id 写回源图层。
  let ind_of = |value: &Value| value.get("ind").and_then(|v| v.as_f64());
  let slot_by_ind = |ind: f64| -> Option<usize> {
    raw.iter()
      .position(|layer| ind_of(layer) == Some(ind))
      .and_then(|raw_index| entry_of_raw[raw_index])
  };
  for idx in 0..raw.len() {
    let tt = raw[idx].get("tt").and_then(|v| v.as_f64()).unwrap_or(0.0) as i64;
    if tt == 0 || !(1..=4).contains(&tt) {
      continue;
    }
    let Some(mask_slot) = entry_of_raw[idx] else {
      continue;
    };
    if out[mask_slot].group != 0 {
      continue;
    }
    let src_slot = match raw[idx].get("tp").and_then(|v| v.as_f64()) {
      Some(tp) => match slot_by_ind(tp) {
        Some(slot) => slot,
        None => continue,
      },
      None => {
        let ind = ind_of(&raw[idx]);
        match ind.and_then(|value| slot_by_ind(value - 1.0)) {
          Some(slot) => slot,
          None => match idx.checked_sub(1).and_then(|prev| entry_of_raw[prev]) {
            Some(slot) => slot,
            None => continue,
          },
        }
      }
    };
    if src_slot == mask_slot {
      continue;
    }
    let gid = ctx.next_group();
    out[src_slot].role = 1;
    out[mask_slot].group = gid;
    out[mask_slot].role = 2;
    out[mask_slot].matte_mode = tt as u32;
    out[mask_slot].matte_src = Some(src_slot);
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
  /// 轨道遮罩分组 id，0 表示普通批次
  group: u32,
  /// 外层轨道遮罩分组 id（嵌套遮罩），0 表示没有外层分组
  parent: u32,
  matte: u32,
  role: u32,
  image: Option<usize>,
  /// 所属图层 pass（模糊 / 蒙版图层及其预合成子图层共享同一 id）
  pass_id: u32,
  /// 所属图层的混合模式（0 正常）
  blend: u32,
  /// 所属图层在编译结果中的下标，前端据此把同一图层的批次归为一组
  layer: u32,
}

/// 一个需要离屏合成的图层 pass（模糊或蒙版）
struct PassRec {
  id: u32,
  sigma_x: f32,
  sigma_y: f32,
  wrap: bool,
  mask_first: usize,
  mask_count: usize,
  first_batch: usize,
  batch_end: usize,
  /// 外层 pass（预合成子图层继承的 pass），用于扩展批次区间
  parent: u32,
  /// 该 pass 所属图层的轨道遮罩分组（0 表示不属于分组）
  group: u32,
}

/// 一条图层蒙版的渲染指令
struct MaskRec {
  /// 0 加 / 1 减 / 2 交
  op: u32,
  inv: bool,
  /// 蒙版覆盖几何所在批次
  batch: usize,
  pass: u32,
  opacity: f32,
  /// 反向「加」蒙版需要先铺满全屏的批次
  quad: Option<usize>,
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
  passes: Vec<PassRec>,
  pass_index: HashMap<u32, usize>,
  masks: Vec<MaskRec>,
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
      passes: Vec::new(),
      pass_index: HashMap::new(),
      masks: Vec::new(),
    }
  }

  /// 预合成 / 蒙版图层是否需要独立离屏 pass
  fn layer_pass(&self, i: usize) -> Option<u32> {
    let layer = &self.c.layers[i];
    if layer.blur.is_some() || !layer.masks.is_empty() {
      Some(i as u32)
    } else {
      None
    }
  }

  /// 登记一个 pass（首次出现时按当前帧的模糊参数算出画布像素 sigma）
  fn ensure_pass(&mut self, id: u32, parent: u32, group: u32) {
    if self.pass_index.contains_key(&id) {
      return;
    }
    let local = self.frame - self.c.layers[id as usize].time_offset;
    let (blurriness, dimensions, edge) = match &self.c.layers[id as usize].blur {
      Some(b) => (
        prop_f32(Some(&b.blurriness), local, 0.0),
        prop_f32(Some(&b.dimensions), local, 1.0) as i32,
        prop_f32(Some(&b.edge), local, 0.0) as i32,
      ),
      None => (0.0, 1i32, 0),
    };
    let world = self.world_of(id as usize);
    let sx = self.c.width / self.c.doc_w;
    let sy = self.c.height / self.c.doc_h;
    // lottie：sigma = 0.3 * Blurriness（图层局部坐标），再乘以世界缩放与画布缩放
    let k = 0.3 * blurriness * world.scale_factor();
    let sigma_x = if dimensions == 3 { 0.0 } else { k * sx };
    let sigma_y = if dimensions == 2 { 0.0 } else { k * sy };
    let index = self.passes.len();
    self.passes.push(PassRec {
      id,
      sigma_x,
      sigma_y,
      wrap: edge == 1,
      mask_first: usize::MAX,
      mask_count: 0,
      // 区间由 add_batch 按实际批次回填：pass 刚创建时可能还没有任何批次
      first_batch: usize::MAX,
      batch_end: usize::MAX,
      parent,
      group,
    });
    self.pass_index.insert(id, index);
  }

  /// 推入一个批次，并把区间扩展到其所有祖先 pass
  fn add_batch(&mut self, batch: Batch) {
    let index = self.batches.len();
    let mut p = batch.pass_id;
    while p != PASS_NONE {
      let Some(&pi) = self.pass_index.get(&p) else {
        break;
      };
      let rec = &mut self.passes[pi];
      if rec.first_batch == usize::MAX {
        rec.first_batch = index;
      }
      rec.batch_end = index + 1;
      p = rec.parent;
    }
    self.batches.push(batch);
  }

  /// 把图层蒙版烘焙成覆盖几何，并登记蒙版记录
  fn emit_masks(&mut self, i: usize, pass: u32, group: u32, role: u32, matte: u32, parent: u32) {
    let (masks, offset) = {
      let layer = &self.c.layers[i];
      (layer.masks.clone(), layer.time_offset)
    };
    if masks.is_empty() {
      return;
    }
    let world = self.world_of(i);
    let c = self.c;
    let cmat = Mat::scale(c.width / c.doc_w, c.height / c.doc_h).mul(&world);
    let local = self.frame - offset;
    let mut first = usize::MAX;
    let mut count = 0usize;
    for mask in &masks {
      if mask.mode == 3 {
        continue;
      }
      let Some(data) = path_data_at(&mask.path, local) else {
        continue;
      };
      let opacity = prop_f32(Some(&mask.opacity), local, 100.0) / 100.0;
      if opacity <= 0.001 {
        continue;
      }
      let Some(path) = bezier_path(&data.verts, &data.ins, &data.outs, data.closed, &cmat) else {
        continue;
      };
      let mut geoms: Vec<Geom> = Vec::new();
      tessellate_fill(&[path], FillRule::NonZero, 0, &mut geoms);
      let mut batch_index = None;
      for geom in &geoms {
        if geom.verts.is_empty() {
          continue;
        }
        let vert_first = self.vertices.len();
        self.vertices.extend_from_slice(&geom.verts);
        batch_index.get_or_insert(self.batches.len());
        self.add_batch(Batch {
          kind: KIND_MASK,
          first: vert_first,
          count: geom.verts.len(),
          color: [1.0, 1.0, 1.0, 1.0],
          alpha: opacity,
          p0: [0.0, 0.0],
          p1: [0.0, 0.0],
          stops: Vec::new(),
          uv_first: 0,
          group,
          parent,
          matte,
          role,
          image: None,
          pass_id: pass,
          blend: 0,
          layer: i as u32,
        });
      }
      let Some(batch) = batch_index else {
        continue;
      };
      // 反向 + 加：先用全屏覆盖铺底，再用该路径做减法才能得到补集
      let quad = if mask.inv && mask.mode == 0 {
        let vert_first = self.vertices.len();
        self.vertices.extend_from_slice(&[
          [0.0, 0.0],
          [c.width, 0.0],
          [c.width, c.height],
          [0.0, 0.0],
          [c.width, c.height],
          [0.0, c.height],
        ]);
        let index = self.batches.len();
        self.add_batch(Batch {
          kind: KIND_FULL,
          first: vert_first,
          count: 6,
          color: [1.0, 1.0, 1.0, 1.0],
          alpha: 1.0,
          p0: [0.0, 0.0],
          p1: [0.0, 0.0],
          stops: Vec::new(),
          uv_first: 0,
          group,
          parent,
          matte,
          role,
          image: None,
          pass_id: pass,
          blend: 0,
          layer: i as u32,
        });
        Some(index)
      } else {
        None
      };
      if first == usize::MAX {
        first = self.masks.len();
      }
      count += 1;
      self.masks.push(MaskRec {
        op: mask.mode,
        inv: mask.inv,
        batch,
        pass,
        opacity,
        quad,
      });
    }
    if count > 0 {
      if let Some(&pi) = self.pass_index.get(&pass) {
        self.passes[pi].mask_first = first;
        self.passes[pi].mask_count = count;
      }
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

  fn render_list(
    &mut self,
    list: &[usize],
    group: u32,
    role: u32,
    matte: u32,
    pass: u32,
    parent: u32,
  ) {
    // 继承分组（预合成子图层）：整段按同一遮罩角色绘制；
    // 但子图层自身还可能构成另一组轨道遮罩（嵌套），需要按内层分组单独发射。
    if group != 0 {
      for &i in list.iter().rev() {
        let layer = &self.c.layers[i];
        if layer.role == 1 {
          // 内层遮罩源在对应的被遮罩图层处统一绘制，保证批次连续
          continue;
        }
        if !self.active(i) {
          continue;
        }
        if layer.role == 2 && layer.group != 0 {
          let group_id = layer.group;
          let matte_mode = layer.matte_mode;
          let matte_src = layer.matte_src;
          if !self.emitted.contains(&group_id) {
            self.emitted.insert(group_id);
            if let Some(j) = matte_src {
              self.render_entry(j, group_id, 1, 0, pass, group);
            }
          }
          self.render_entry(i, group_id, 2, matte_mode, pass, group);
        } else {
          self.render_entry(i, group, role, matte, pass, parent);
        }
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
        let group_id = layer.group;
        let matte_mode = layer.matte_mode;
        let matte_src = layer.matte_src;
        if !self.emitted.contains(&group_id) {
          self.emitted.insert(group_id);
          // 同一个遮罩源可被多个被遮罩图层复用，按分组逐个绘制
          if let Some(j) = matte_src {
            self.render_entry(j, group_id, 1, 0, pass, 0);
          }
        }
        self.render_entry(i, group_id, 2, matte_mode, pass, 0);
      } else {
        self.render_entry(i, 0, 0, 0, pass, 0);
      }
    }
  }

  fn render_entry(
    &mut self,
    i: usize,
    group: u32,
    role: u32,
    matte: u32,
    pass: u32,
    parent: u32,
  ) {
    if !self.active(i) {
      return;
    }
    let world = self.world_of(i);
    let alpha = self.alpha_of(i);
    if alpha <= 0.001 {
      return;
    }
    let kind = self.c.layers[i].kind;
    // 图层自身需要模糊 / 蒙版时开一个新的 pass，否则继承父 pass
    let my_pass = self.layer_pass(i).unwrap_or(pass);
    if my_pass != PASS_NONE {
      self.ensure_pass(my_pass, pass, group);
    }
    if kind == KIND_PRECOMP {
      let children = self.c.layers[i].children.clone();
      self.render_list(&children, group, role, matte, my_pass, parent);
      self.emit_masks(i, my_pass, group, role, matte, parent);
      return;
    }
    if kind == KIND_NULL {
      return;
    }
    self.emit(i, world, alpha, group, role, matte, my_pass, parent);
    self.emit_masks(i, my_pass, group, role, matte, parent);
  }

  fn emit(
    &mut self,
    i: usize,
    world: Mat,
    layer_alpha: f32,
    group: u32,
    role: u32,
    matte: u32,
    pass: u32,
    parent: u32,
  ) {
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
      let (p, g, _) = walk_items(&shapes, Mat::identity(), 1.0, local);
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
        self.add_batch(Batch {
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
          parent,
          matte,
          role,
          image: Some(image),
          pass_id: pass,
          blend: layer.blend,
          layer: i as u32,
        });
        continue;
      }
      let Some(paint) = paints.get(geom.paint) else {
        continue;
      };
      let g0 = world.apply(paint.p0);
      let g1 = world.apply(paint.p1);
      self.add_batch(Batch {
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
        parent,
        matte,
        role,
        image: None,
        pass_id: pass,
        blend: layer.blend,
        layer: i as u32,
      });
    }
  }
}

fn render_frame(c: &Compiled, frame: f32) -> Vec<u8> {
  let mut r = Renderer::new(c, frame);
  let roots = c.roots.clone();
  r.render_list(&roots, 0, 0, 0, PASS_NONE, 0);

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
    words.push(b.pass_id);
    words.push(b.blend);
    words.push(b.layer);
    // 其后是 word 61：外层轨道遮罩分组（嵌套遮罩），剩余字保留
    words.push(b.parent);
    let used = 13 + MAX_STOPS * 5 + 9;
    for _ in used..BATCH_WORDS {
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

  // 尾部：图层 pass（模糊）与蒙版记录，供 JS 做离屏合成
  let active_passes: Vec<&PassRec> = r
    .passes
    .iter()
    .filter(|p| p.first_batch != usize::MAX && p.batch_end > p.first_batch)
    .collect();
  words.push(active_passes.len() as u32);
  for p in &active_passes {
    words.push(p.id);
    words.push(p.first_batch as u32);
    words.push((p.batch_end - p.first_batch) as u32);
    words.push(if p.wrap { 1 } else { 0 });
    words.push(p.sigma_x.to_bits());
    words.push(p.sigma_y.to_bits());
    // word 6：该 pass 所在图层的轨道遮罩分组（0 表示不在分组内）
    words.push(p.group);
    words.push(if p.mask_count > 0 {
      p.mask_first as u32
    } else {
      u32::MAX
    });
    words.push(p.mask_count as u32);
    words.push(0);
  }
  words.push(r.masks.len() as u32);
  for m in &r.masks {
    words.push(m.op);
    words.push(if m.inv { 1 } else { 0 });
    words.push(m.batch as u32);
    words.push(m.pass);
    words.push(m.opacity.to_bits());
    words.push(m.quad.map(|q| q as u32).unwrap_or(u32::MAX));
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
  fn merge_paths_keeps_hole_via_fill_rule() {
    // 外框顺时针、内框逆时针：mm=1 合并成一条路径后按 nonzero 填充，
    // 面积应为「外框 - 内框」；若不合并而逐条三角化，内框会被单独填满。
    let shapes = serde_json::json!([{
      "ty": "gr",
      "it": [
        {"ty": "sh", "ks": {"a": 0, "k": {
          "c": true,
          "v": [[-100, -60], [100, -60], [100, 60], [-100, 60]],
          "i": [[0, 0], [0, 0], [0, 0], [0, 0]],
          "o": [[0, 0], [0, 0], [0, 0], [0, 0]]
        }}},
        {"ty": "sh", "ks": {"a": 0, "k": {
          "c": true,
          "v": [[-30, -20], [-30, 20], [30, 20], [30, -20]],
          "i": [[0, 0], [0, 0], [0, 0], [0, 0]],
          "o": [[0, 0], [0, 0], [0, 0], [0, 0]]
        }}},
        {"ty": "mm", "mm": 1},
        {"ty": "fl", "r": 1, "c": {"a": 0, "k": [1, 0, 0, 1]}, "o": {"a": 0, "k": 100}},
        {"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}}
      ]
    }]);
    let doc = serde_json::json!({
      "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
      "layers": [base_layer(shapes)]
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
        area += ((p1[0] - a[0]) * (p2[1] - a[1]) - (p2[0] - a[0]) * (p1[1] - a[1])).abs() * 0.5;
      }
    }
    // 外框 200x120 = 24000，内框 60x40 = 2400
    let expected = 24000.0 - 2400.0;
    let ratio = area / expected;
    assert!(ratio > 0.98 && ratio < 1.02, "合并后面积 {area}，面积比 {ratio}");
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

  /// 尾部（pass + mask 记录）的起始字下标
  fn tail_start(words: &[u32]) -> usize {
    3 + words[0] as usize * BATCH_WORDS + words[1] as usize * 2 + words[2] as usize * 2
  }

  /// 三角化后所有批次的总面积（像素）
  fn total_area(words: &[u32]) -> f32 {
    let batch_count = words[0] as usize;
    let base = vertex_base(words);
    let mut area = 0.0f32;
    for b in 0..batch_count {
      let header = 3 + b * BATCH_WORDS;
      let vertex_start = words[header + 1] as usize;
      let vertex_count = words[header + 2] as usize;
      for tri in (0..vertex_count).step_by(3) {
        let a = read_vertex(words, base, vertex_start + tri);
        let p1 = read_vertex(words, base, vertex_start + tri + 1);
        let p2 = read_vertex(words, base, vertex_start + tri + 2);
        area += (((p1[0] - a[0]) * (p2[1] - a[1]) - (p2[0] - a[0]) * (p1[1] - a[1])).abs()) * 0.5;
      }
    }
    area
  }

  #[test]
  fn gaussian_blur_emits_layer_pass() {
    let mut layer = base_layer(serde_json::json!([rect_group(
      serde_json::json!({"ty": "fl", "c": {"a": 0, "k": [1, 0, 0, 1]}, "o": {"a": 0, "k": 100}})
    )]));
    // lottie 的 Gaussian Blur（ef ty=29）：Blurriness 20、Dimensions 1、Edge 0
    layer["ef"] = serde_json::json!([{
      "ty": 29, "en": 1, "nm": "Gaussian Blur",
      "ef": [
        {"v": {"a": 0, "k": 20}},
        {"v": {"a": 0, "k": 1}},
        {"v": {"a": 0, "k": 0}}
      ]
    }]);
    let doc = serde_json::json!({
      "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
      "layers": [layer]
    });
    let compiled = compile_doc(&doc, 400.0, 400.0).expect("compile");
    let words = read_words(&render_frame(&compiled, 0.0));
    let tail = tail_start(&words);
    assert_eq!(words[tail], 1, "模糊图层应登记一个离屏 pass");
    let p = tail + 1;
    // sigma = 0.3 × Blurriness × 世界缩放 × 画布缩放（这里都为 1）
    let sigma_x = f32::from_bits(words[p + 4]);
    let sigma_y = f32::from_bits(words[p + 5]);
    assert!((sigma_x - 6.0).abs() < 0.01, "sigmaX={sigma_x}");
    assert!((sigma_y - 6.0).abs() < 0.01, "sigmaY={sigma_y}");
    assert_eq!(words[p + 3], 0, "Edge Mode 0 应为 clamp");
    assert_eq!(words[p + 8], 0, "该图层没有蒙版");
    // 内容批次应归属该 pass，JS 才会走离屏渲染
    let batch_pass = words[3 + 58] as usize;
    assert_eq!(batch_pass, words[p] as usize, "批次应带上 passId");
  }

  #[test]
  fn layer_mask_emits_coverage_and_inverse_quad() {
    let make = |inv: bool| {
      let mut layer = base_layer(serde_json::json!([rect_group(
        serde_json::json!({"ty": "fl", "c": {"a": 0, "k": [0, 1, 0, 1]}, "o": {"a": 0, "k": 100}})
      )]));
      layer["masksProperties"] = serde_json::json!([{
        "mode": "a", "inv": inv, "nm": "m",
        "pt": {"a": 0, "k": {
          "c": true,
          "v": [[-50, -40], [50, -40], [50, 40], [-50, 40]],
          "i": [[0, 0], [0, 0], [0, 0], [0, 0]],
          "o": [[0, 0], [0, 0], [0, 0], [0, 0]]
        }},
        "o": {"a": 0, "k": 100}
      }]);
      serde_json::json!({
        "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
        "layers": [layer]
      })
    };
    let plain = read_words(&render_frame(&compile_doc(&make(false), 400.0, 400.0).unwrap(), 0.0));
    let tail = tail_start(&plain);
    assert_eq!(plain[tail], 1, "蒙版图层应登记一个离屏 pass");
    let mask_start = tail + 1 + plain[tail] as usize * 10;
    assert_eq!(plain[mask_start], 1, "应登记一条蒙版记录");
    let m = mask_start + 1;
    assert_eq!(plain[m], 0, "mode a → 加");
    assert_eq!(plain[m + 1], 0, "非反向");
    assert_eq!(plain[m + 5], u32::MAX, "非反向「加」不需要铺底 quad");
    // 蒙版覆盖几何是 kind=5 批次
    let mut has_coverage = false;
    for b in 0..plain[0] as usize {
      if plain[3 + b * BATCH_WORDS] == KIND_MASK {
        has_coverage = true;
      }
    }
    assert!(has_coverage, "缺少蒙版覆盖几何（kind=5）");

    // 反向「加」必须额外铺一层全屏覆盖（kind=6），否则补集算不出来
    let inverted = read_words(&render_frame(&compile_doc(&make(true), 400.0, 400.0).unwrap(), 0.0));
    let tail = tail_start(&inverted);
    let mask_start = tail + 1 + inverted[tail] as usize * 10;
    let m = mask_start + 1;
    assert_eq!(inverted[m + 1], 1, "应为反向蒙版");
    assert_ne!(inverted[m + 5], u32::MAX, "反向「加」应带全屏 quad 批次");
    let quad = inverted[m + 5] as usize;
    assert_eq!(inverted[3 + quad * BATCH_WORDS], KIND_FULL, "quad 应为 kind=6");
  }

  #[test]
  fn round_corner_star_and_repeater_shapes_render() {
    // rd：200×120 矩形四角各切半径 20 → 面积减少 (4-π)·r²
    let rounded = {
      let items = serde_json::json!([
        {"ty": "rc", "p": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [200, 120]}, "r": {"a": 0, "k": 0}},
        {"ty": "rd", "r": {"a": 0, "k": 20}, "nm": "rd"},
        {"ty": "fl", "c": {"a": 0, "k": [1, 0, 0, 1]}, "o": {"a": 0, "k": 100}},
        {"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}}
      ]);
      serde_json::json!({
        "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
        "layers": [base_layer(serde_json::json!([{"ty": "gr", "it": items, "nm": "g"}]))]
      })
    };
    let words = read_words(&render_frame(&compile_doc(&rounded, 400.0, 400.0).unwrap(), 0.0));
    let expected = 200.0 * 120.0 - (4.0 - std::f32::consts::PI) * 20.0 * 20.0;
    let ratio = total_area(&words) / expected;
    assert!(ratio > 0.97 && ratio < 1.03, "圆角后面积比 {ratio}");

    // sr：5 角星（外径 100、内径 50）面积 = 5·R·r·sin(π/5)
    let star = serde_json::json!({
      "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
      "layers": [base_layer(serde_json::json!([{
        "ty": "gr",
        "it": [
          {"ty": "sr", "sy": 1, "d": 1, "pt": {"a": 0, "k": 5}, "p": {"a": 0, "k": [0, 0]},
           "r": {"a": 0, "k": 0}, "ir": {"a": 0, "k": 50}, "is": {"a": 0, "k": 0},
           "or": {"a": 0, "k": 100}, "os": {"a": 0, "k": 0}},
          {"ty": "fl", "c": {"a": 0, "k": [0, 0, 1, 1]}, "o": {"a": 0, "k": 100}},
          {"ty": "tr", "p": {"a": 0, "k": [0, 0]}, "a": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0}, "o": {"a": 0, "k": 100}}
        ]
      }]))]
    });
    let words = read_words(&render_frame(&compile_doc(&star, 400.0, 400.0).unwrap(), 0.0));
    let expected = 5.0 * 100.0 * 50.0 * (std::f32::consts::PI / 5.0).sin();
    let ratio = total_area(&words) / expected;
    assert!(ratio > 0.95 && ratio < 1.05, "星形面积比 {ratio}");

    // rp：3 个副本 → 顶点数约为单个副本的 3 倍
    let make = |repeat: bool| {
      let mut items = vec![
        serde_json::json!({"ty": "rc", "p": {"a": 0, "k": [0, 0]}, "s": {"a": 0, "k": [40, 40]}, "r": {"a": 0, "k": 0}}),
        serde_json::json!({"ty": "fl", "c": {"a": 0, "k": [0, 0, 0, 1]}, "o": {"a": 0, "k": 100}}),
      ];
      if repeat {
        items.push(serde_json::json!({
          "ty": "rp", "c": {"a": 0, "k": 3}, "o": {"a": 0, "k": 0}, "m": 1, "nm": "rp",
          "tr": {
            "p": {"a": 0, "k": [50, 0]}, "a": {"a": 0, "k": [0, 0]},
            "s": {"a": 0, "k": [100, 100]}, "r": {"a": 0, "k": 0},
            "so": {"a": 0, "k": 100}, "eo": {"a": 0, "k": 100}
          }
        }));
      }
      serde_json::json!({
        "v": "5.7.4", "fr": 30, "ip": 0, "op": 60, "w": 400, "h": 400,
        "layers": [base_layer(serde_json::json!([{"ty": "gr", "it": items, "nm": "g"}]))]
      })
    };
    let single = read_words(&render_frame(&compile_doc(&make(false), 400.0, 400.0).unwrap(), 0.0));
    let triple = read_words(&render_frame(&compile_doc(&make(true), 400.0, 400.0).unwrap(), 0.0));
    let ratio = triple[1] as f32 / single[1] as f32;
    assert!((ratio - 3.0).abs() < 0.05, "重复器顶点数比 {ratio}");
    // 副本沿 x 平移：最右端应明显大于单副本
    let base = vertex_base(&triple);
    let mut max_x = f32::MIN;
    for i in 0..triple[1] as usize {
      max_x = max_x.max(read_vertex(&triple, base, i)[0]);
    }
    assert!(max_x > 240.0, "重复器平移未生效：max_x={max_x}");
  }
}
