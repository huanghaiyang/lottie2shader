# lottie2shader

把 Lottie 形状动画编译成 GPU 三角面片渲染计划：Rust 侧完成 JSON 解析、路径求值、贝塞尔离散化与三角化，浏览器端每帧只做矩阵变换与顶点输出，由 WebGL2 完成光栅化、渐变与合成。

## 设计目标

- 不依赖 `wasm-bindgen`：以裸 WASM 导出函数 + 线性内存交换数据，前端直接 `WebAssembly.instantiate` 即可加载。
- 几何在编译期一次性烘焙到图层局部坐标，运行时不再解析 JSON。
- 每帧只输出「批次头 + 顶点」，前端按批次 `drawArrays`。
- 体积优先：`opt-level = "z"`、LTO、`panic = "abort"`、`strip = true`。

## 支持范围

当前覆盖形状动画与常见特效：

| 类别 | 支持项 |
| --- | --- |
| 图层 | 形状（`ty=4`）、纯色（`ty=1`）、图片（`ty=2`）、空图层（`ty=3`）、预合成（`ty=0`）、3D 相机（`ty=13`，仅顶层 comp） |
| 形状 | 矩形 `rc`（含圆角）、椭圆 `el`、贝塞尔 `sh`、组 `gr`/`tr`、trim path `tm`、合并路径 `mm`（1/2/5，Exclude 用绕向反转实现） |
| 画笔 | 纯色填充 `fl`、纯色描边 `st`、线性/径向渐变填充 `gf`、渐变描边 `gs` |
| 蒙版 | `masksProperties`（a/s/i/n）、扩张/收缩 `x`（轮廓法线偏移）、轨道遮罩（alpha / 反相 alpha） |
| 特效 | 高斯模糊 `ef ty=29`，多层模糊按方差可加合并为一次等效 sigma（`σ=√Σσᵢ²`） |
| 3D | 图层 `rx/ry/rz` + 透视；有相机时 parent/container 链统一走 4×4 世界矩阵，投影后补偿 comp 中心 |
| 变换 | 位置 / 锚点 / 缩放 / 旋转 / 不透明度关键帧与缓动，父子关系与预合成嵌套 |

暂不支持（显式报错并由前端回退 canvas，不静默画错）：文本图层（需字体轮廓）、未烘焙的表达式、非高斯图层特效（发光/投影/色阶等）、亮度轨道遮罩、星形/重复器等形状修饰器、合并路径的 `mm=3 Subtract` / `mm=4 Intersect`、预合成内部的 3D 相机。

## 目录结构

```
src/lib.rs              核心实现与单元测试
scripts/check.mjs       用 Node 加载 wasm、跑样例动画的冒烟脚本
scripts/verify-demos.mjs 用 Node 校验指定的单个 Lottie JSON（构造特性数据用）
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

冒烟脚本可用环境变量 `L2S_WASM` 指定其他 wasm 路径。`scripts/verify-demos.mjs` 用于校验构造出来的单个 JSON（例如多模糊叠加、3D 相机这类真实样例里没有的特性）：

```powershell
$env:L2S_WASM="D:\sgames\3d-slide\src\lib\lottieShader\lottie2shader.wasm"
$env:L2S_FILES="D:/path/camera-demo.json;D:/path/blur-stack-demo.json"
node scripts/verify-demos.mjs
```

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
2. `vertexCount`（浮点对 `x, y` 的个数）
3. `uvCount`（浮点对 `u, v` 的个数）
4. `batchCount` 个批次头，每个 `BATCH_WORDS = 104` 字
5. 共享顶点数组（`f32 x, y` 成对，位模式存放）
6. 共享 UV 数组（`f32 u, v` 成对）
7. 尾部：pass 记录（模糊离屏合成）与 mask 记录（蒙版覆盖几何）

批次头字段顺序（前 13 字 + 16×5 色标 + 9 字，其余补零）：

```
kind, first, count,
color.r, color.g, color.b, color.a,
alpha,
p0.x, p0.y, p1.x, p1.y,
stopCount,
16 × (offset, r, g, b, a)   // 不足补零
uvFirst, group, matte, role, image, passId, blend, layer, parent
```

`kind`：`0` 纯色、`1` 线性渐变、`2` 径向渐变、`3` 图片、`5` 蒙版覆盖、`6` 全屏四边形；`first` / `count` 为共享顶点数组中的起始索引与顶点数，前端按此区间 `drawArrays`。`image` 为图片资源下标（`u32::MAX` 表示非图片），`passId` 指向尾部的离屏 pass 记录（`u32::MAX` 表示直接绘制）。

pass 记录（10 字）：`id, firstBatch, batchCount, wrap, sigmaX, sigmaY, group, maskFirst, maskCount, 0`，其中 `sigmaX/sigmaY` 是已合并的画布像素 σ，`wrap` 表示边缘模式为 repeat。mask 记录（6 字）描述每条蒙版的运算与覆盖批次。所有顶点已按目标画布缩放，坐标即渲染坐标。

## 依赖

- [serde_json](https://crates.io/crates/serde_json)：Lottie JSON 解析
- [lyon](https://crates.io/crates/lyon)：路径离散化与三角化
