# lottie2shader

把 Lottie 形状动画编译成 GPU 三角面片渲染计划：Rust 侧完成 JSON 解析、路径求值、贝塞尔离散化与三角化，浏览器端每帧只做矩阵变换与顶点输出，由 WebGL2 完成光栅化、渐变与合成。

## 设计目标

- 不依赖 `wasm-bindgen`：以裸 WASM 导出函数 + 线性内存交换数据，前端直接 `WebAssembly.instantiate` 即可加载。
- 几何在编译期一次性烘焙到图层局部坐标，运行时不再解析 JSON。
- 每帧只输出「批次头 + 顶点」，前端按批次 `drawArrays`。
- 体积优先：`opt-level = "z"`、LTO、`panic = "abort"`、`strip = true`。

## 支持范围

当前覆盖形状动画 MVP：

| 类别 | 支持项 |
| --- | --- |
| 图层 | 形状图层（`ty=4`）、嵌套组（`gr` / `tr`）、图层父子关系（`parent`） |
| 形状 | 矩形 `rc`（含圆角）、椭圆 `el`、贝塞尔路径 `sh` |
| 画笔 | 纯色填充 `fl`、纯色描边 `st`、线性/径向渐变填充 `gf`（`gs` 取首色标近似） |
| 变换 | 位置 / 锚点 / 缩放 / 旋转 / 不透明度关键帧与缓动 |

暂不支持：路径形变关键帧（取首帧）、文本、图片、遮罩、表达式、3D 图层。遇到不支持的图层前端可自动回退到原始 Lottie 渲染。

## 目录结构

```
src/lib.rs              核心实现与单元测试
scripts/check.mjs       用 Node 加载 wasm、跑样例动画的冒烟脚本
public/animations/      Lottie 样例 JSON
dist/lottie2shader.wasm 构建产物
```

## 构建

需要 `wasm32-unknown-unknown` 目标：

```bash
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown
copy target\wasm32-unknown-unknown\release\lottie2shader.wasm dist\lottie2shader.wasm
```

## 测试

```bash
cargo test                 # Rust 侧单元测试
node scripts/check.mjs     # 冒烟测试：编译并渲染 public/animations 下的样例
```

冒烟脚本可用环境变量 `L2S_WASM` 指定其他 wasm 路径。

## WASM 接口

| 导出函数 | 说明 |
| --- | --- |
| `l2s_alloc(len) -> *mut u8` | 分配输入缓冲区，用于写入 Lottie JSON |
| `l2s_free(ptr, len)` | 释放输入缓冲区 |
| `l2s_compile(ptr, len, w, h) -> u32` | 编译 Lottie，`1` 成功、`0` 失败 |
| `l2s_set_size(w, h)` | 运行时更新输出画布尺寸（不重新三角化） |
| `l2s_render(frame) -> u32` | 生成指定帧的渲染计划，`1` 成功、`0` 失败 |
| `l2s_result_ptr()` / `l2s_result_len()` | 渲染结果的指针与字节长度 |
| `l2s_error_ptr()` / `l2s_error_len()` | 最近一次错误的 UTF-8 消息 |
| `l2s_fps()` / `l2s_ip()` / `l2s_op()` | 动画帧率、起始帧、结束帧 |

典型调用顺序：

```js
const bytes = new TextEncoder().encode(JSON.stringify(lottie));
const ptr = ex.l2s_alloc(bytes.length);
new Uint8Array(ex.memory.buffer, ptr, bytes.length).set(bytes);
const ok = ex.l2s_compile(ptr, bytes.length, width, height);
ex.l2s_free(ptr, bytes.length);
ex.l2s_render(frame);
const words = new Uint32Array(ex.memory.buffer, ex.l2s_result_ptr(), ex.l2s_result_len() / 4);
```

## 渲染计划格式

小端 `u32` 序列：

1. `batchCount`
2. `batchCount` 个批次头，每个 53 字（`13 + 8 × 5`）
3. 共享顶点数组（`f32 x, y` 成对，位模式存放）

批次头字段顺序：

```
kind, first, count,
color.r, color.g, color.b, color.a,
alpha,
p0.x, p0.y, p1.x, p1.y,
stopCount,
8 × (offset, r, g, b, a)   // 不足 8 个补零
```

`kind`：`0` 纯色、`1` 线性渐变、`2` 径向渐变；`first` / `count` 为共享顶点数组中的起始索引与顶点数，前端按此区间 `drawArrays`。所有顶点已按目标画布缩放，坐标即渲染坐标。

## 依赖

- [serde_json](https://crates.io/crates/serde_json)：Lottie JSON 解析
- [lyon](https://crates.io/crates/lyon)：路径离散化与三角化
