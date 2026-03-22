# PlaneDetect

## 输入点云格式（`plane_test_node`）

- **传输**：`sensor_msgs/PointCloud2`，由节点内 **Raw 数据直传 GPU** 解包（`processRawMsg` / `unpackROSMsgKernel`），**不**先转为 `pcl::PointCloud` 再上传，以保证低延迟。
- **必选**：字段 `x`, `y`, `z`。
- **可选强度**：`intensity`（优先）；若无则使用 `reflectivity`；若均无，内部 `GPUPoint3f.intensity` 为 0，输出点云 `PointXYZI` 中 `intensity` 亦为 0。
- **颜色**：`rgb`、`r`/`g`/`b`、`rgba` 等 **不参与解包**（常见 XYZRGB 布局仍可处理，仅几何用于检测）。

详见 [`config/plane_detection.yaml`](config/plane_detection.yaml) 中 `input_topic` 段注释。

调试时可提高日志级别以查看每帧强度字段来源：`rosrun ... __log_level:=debug` 或 `ROSCONSOLE_CONFIG_FILE`。

### 冒烟（布局）

离线校验与节点相同的字段规则（XYZI / XYZRGB 风格 / reflectivity）：

```bash
source /opt/ros/noetic/setup.bash   # 或你的 ROS 发行版
python3 $(rospack find PlaneDetect)/scripts/smoke_pointcloud2_layouts.py
```

全量编译：`catkin_make`（工作空间根目录）。
