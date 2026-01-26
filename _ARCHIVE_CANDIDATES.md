# 待归档文件清单（✅ 已完成归档）

**归档日期**: 2026-01-24
**编译验证**: ✅ 100% 成功

## 📋 分析说明

本清单基于以下分析：
1. **CMakeLists.txt 依赖分析**：扫描所有包的编译配置
2. **Include 依赖分析**：搜索所有 #include 语句
3. **文件存在性检查**：确认文件是否为空或未使用

## ⚠️ 安全保护

**以下核心文件将被保护，不会移动**：
- ✅ `src/PlaneDetect/src/PlaneDetect.cpp` / `.cu`
- ✅ `src/gpu_demo/src/QuadricDetect.cpp` / `.cu`
- ✅ `src/gpu_demo/src/GPUPreprocessor.cpp` / `GPUPreprocessor_kernels.cu`
- ✅ 所有相关的头文件 (`.h`, `.cuh`)

---

## 📦 待归档文件分类

### 1️⃣ 空文件（完全无用，可安全移动）

这些文件存在但内容为空，未被任何地方引用：

```
src/point_cloud_generator/src/batch_ransac_test.cpp
src/point_cloud_generator/src/QuadricDetector.cpp
src/point_cloud_generator/src/QuadricDetector_GPU.cu
src/point_cloud_generator/include/point_cloud_generator/QuadricDetector.h
src/point_cloud_generator/src/gpu_kernels.cu
src/point_cloud_generator/src/gpu_kernels.cuh
src/super_voxel/src/test_library.cpp
```

**说明**：
- `test_library.cpp` 虽然被 launch 文件引用，但文件本身为空，无法编译
- 其他文件完全未被引用

---

### 2️⃣ 未编译的源文件（备份/占位符）

这些文件存在但未在 CMakeLists.txt 中编译：

```
src/super_voxel/src/supervoxel copy.cpp
src/gpu_demo/src/test_main.cpp
```

**说明**：
- `supervoxel copy.cpp` 是备份文件（文件名包含空格）
- `test_main.cpp` 只是一个占位符程序，未在 CMakeLists.txt 中编译

---

### 3️⃣ 未使用的目录（整个目录可移动）

```
src/supervoxel_demo/
```

**说明**：
- 该目录包含头文件但没有任何 CMakeLists.txt
- 未被任何其他包引用
- 包含文件：
  - `include/supervoxel_demo/supervoxel_gpu_simple.cuh`
  - `include/supervoxel_demo/supervoxel_gpu.cuh`
  - `include/supervoxel_demo/supervoxel_gpu.h`

---

### 4️⃣ 文档和图片文件（可选归档）

这些文件不影响编译，但可以归档以保持项目整洁：

```
bspline_tex.pdf                          # 根目录，PDF文档
新的凸包图.png                            # 根目录，图片文件
src/super_voxel/2025-09-26 19-48-17 的屏幕截图.png  # 截图文件
```

**建议**：保留文档文件（如 README.md, SUPERVOXEL_INTEGRATION.md），仅归档图片和PDF

---

### 5️⃣ 已禁用但保留的文件（⚠️ 不移动）

以下文件虽然被禁用，但 CMakeLists.txt 中有注释说明，建议保留：

```
src/gpu_demo/src/demo_test.cpp.disabled
```

**说明**：CMakeLists.txt 第135-146行有注释说明该文件已禁用但代码保留

---

## 📊 统计信息

- **空文件**: 7 个
- **未编译源文件**: 2 个
- **未使用目录**: 1 个（包含 3 个头文件）
- **文档/图片**: 3 个
- **总计**: 约 15 个文件/目录

---

## ✅ 确认后执行

确认后，将执行以下操作：

1. 创建 `_archive_legacy/` 目录
2. 保持目录结构移动文件
3. 验证编译（执行 `catkin_make`）
4. 生成归档报告

---

## 🔍 验证方法

移动后，将执行：
```bash
cd /home/ubuntu/demo_quadric
catkin_make
```

确保编译成功率为 100%。
