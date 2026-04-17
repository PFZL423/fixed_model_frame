# 论文写作规划文档
> 期刊目标：IEEE Sensors Journal（Regular Paper，8～10页双栏）
> 更新时间：2026/04/13

---

## 一、整体页面分配

| 章节 | 预计页数 |
|---|---|
| Abstract | ~0.2 页 |
| Introduction | ~1.5 页 |
| Related Work | ~1 页 |
| Method | ~2.5～3 页 |
| Experiments | ~2.5～3 页 |
| Conclusion | ~0.3 页 |
| References | ~0.5～1 页 |

---

## 二、Introduction 写作规划

### 已定稿逻辑链（中文草稿）

```
点云是多传感器通用数据形式（宽背景，不绑定SLAM）
        ↓
现有方法依赖平面假设（有文献支撑的现状描述）
        ↓
平面假设在曲面场景下有本质局限（具体化缺陷）
        ↓
原始点云存储代价高，需要紧凑表征（第二个独立动机）
        ↓
二次曲面同时回应两个问题，但提取效率问题尚未解决
        ↓
现有方法两类局限：隐式表征无限延伸 + RANSAC迭代代价
        ↓
本文提出方法（一句话引出贡献bullet points）
```

### 引用文献缺口（待补充）

| 位置 | 需要的文献类型 |
|---|---|
| 第一段 | 机器人导航综述、自动驾驶感知综述、三维重建综述 |
| 第一段 | LOAM / LeGO-LOAM（里程计代表） |
| 第一段 | 几何原语提取综述 |
| 第一段 | 工业管道检测应用、非结构化场景感知 |
| 第二段 | 点云紧凑表征综述（OctoMap等） |
| 第二段 | 高阶曲面/隐式表征/神经场对比综述（可选） |
| 第三段 | Schnabel 2007（点云原语提取） |

### 已有引用（可直接使用）
- `li2024pssba` — PSS-BA（LiDAR束调整）
- `9813516` — Adaptive Voxel Mapping
- `yuan2023voxelmap` — VoxelMap++
- `deng2026` — 3D场景表征综述
- `10167749` — Xia 2023（Quadric representations）
- `8644023` — Birdal 2020（Minimal Fits）

### 关键修改事项（审稿人攻击点）
1. ✅ 删除 O(N⁹) 错误，改为：内点率为r时迭代次数 k∝r^{-9}
2. ✅ 解耦SLAM，第一句改为通用点云处理背景
3. ✅ 「通常采用」改为「一类代表性方法」
4. ✅ 「维度爆炸」改为「最小采样集规模带来的迭代代价」
5. ✅ 删除贡献段中「解决了」，改为「有效缓解了」
6. ✅ 贡献bullet 1加显式方程适用范围限定语：「对于可在局部对齐坐标系下参数化为z=f(x,y)的曲面」
7. ✅ 过渡段压缩为一句话引出bullet points，细节全移入bullet

---

## 三、Method 章节规划

### 结构（三节对应三个贡献）

```
Section III. Method

  A. Explicit Quadric Parameterization（显式参数化表征）
     - 显式方程 z = ax²+bxy+cy²+dx+ey+f（6个系数）
     - 局部坐标对齐：transform[12]（3×4旋转平移矩阵）
     - 2D凸包边界裁剪：hull_points_local
     - 预计篇幅：~0.8页（含数学推导）

  B. Voxel-Constrained Sampling Strategy（体素局部性约束采样）
     - 体素哈希索引构建
     - 将采样范围限制在局部连续邻域
     - 对有效内点率的提升分析
     - 预计篇幅：~0.5页

  C. Full-Pipeline GPU Acceleration（全链路GPU加速）
     - 批量并行RANSAC（1024模型同时处理）
     - 两阶段粗筛-精选（coarse ratio=2% → top-K精选）
     - 自定义反幂迭代替代cuSolver（针对6×6小矩阵）
     - Raw→GPU零拷贝接口（unpackROSMsgKernel）
     - 预计篇幅：~1页
```

### Method开头必须有系统流程图（约0.3～0.5页）

```
Raw PointCloud
      ↓
GPU Unpack（零拷贝）
      ↓
Voxel Filter（GPU）
      ↓
Plane Detection（批量GPU RANSAC）
      ↓
Remaining Cloud
      ↓
Voxel Hash Indexing → 体素约束采样
      ↓
Batch GPU RANSAC（1024并行，显式二次曲面）
      ↓
Detected Primitives + 2D Convex Hull
```

---

## 四、Experiments 章节规划

### 数据集计划

| 数据集 | 序列数量 | 传感器 | 用途 |
|---|---|---|---|
| TUM RGB-D | 2～3个序列 | Kinect | 公开benchmark，增加可信度 |
| 自录LiDAR | 2个序列 | Velodyne等 | 真实大规模点云验证 |

TUM推荐序列：fr1/desk、fr3/long_office_household（含曲面物体）

### 实验内容与页面分配

| 实验内容 | 预计页数 | 说明 |
|---|---|---|
| 实现细节（参数表） | ~0.3页 | GPU型号、CUDA版本、关键参数 |
| 数据集描述表 | ~0.3页 | 序列名、帧数、传感器、场景特点 |
| 定量对比（vs基准） | ~0.5页 | 对比PCL SACSegmentation（CPU） |
| 消融实验 | ~0.8页 | 见下表，最重要的部分 |
| 定性结果图 | ~0.5～0.8页 | RViz可视化截图处理 |

### 消融实验设计（每条对应一个设计选择）

| 消融变体 | 验证目标 |
|---|---|
| 去掉体素约束 → 全局随机采样 | 证明局部约束有效 |
| 去掉两阶段粗筛 → 仅全量验证 | 证明两阶段加速有效 |
| 显式表征 vs 隐式表征（边界精度） | 证明显式边界优势 |
| batch_size: 1024 vs 256 vs 64 | 证明并行规模选择合理 |

### 评估指标

- **FPS**（主要性能指标，你的核心声称）
- **点到曲面平均残差 RMSE**（拟合精度）
- **检测到的几何原语数量**
- **点云覆盖率**（被检测原语解释的点占总点数比例）

---

## 五、开源代码注意事项

- 投稿前删除所有 `// 🚧待实现` 注释
- 删除 `cusolverDnHandle_t` 残留声明（如确认未使用）
- 清理 `_archive_legacy/` 目录或从仓库中移除
- 确保 LO-RANSAC 实现完整后再声称此贡献
- README 提供环境配置步骤（ROS版本、CUDA版本、依赖）

---

## 六、IEEE Sensors Journal 投稿信息

- 格式：双栏，Regular Paper 8～10页
- 拒稿率：总体约50%，但桌面拒稿占20～30%；真正送审后拒稿率更低
- 典型流程：送审 → 大修 → 小修/录用（2～3轮）
- 一轮直接录用概率：< 5%，属罕见
- 格式拒稿（桌面）：不影响后续重投，无学术记录
