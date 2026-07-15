# GeoReward

利用 Depth Anything 3 (DA3) 的显式几何信息（深度、相机位姿、置信度）作为 Reward 信号，在 Wan2.2 I2V 视频生成模型的推理阶段提升生成视频的物理一致性。场景：机械臂操作。

## 项目结构

```
now/
├── geo_reward/              # 核心模块：GeoReward计算 + BoN流程
│   ├── da3_reward.py        # DA3GeoReward（投射一致性+锚定+置信度）
│   ├── bon_pipeline.py      # GeoRewardBoN / GeoRewardBoNOffline
│   └── utils.py             # 格式转换、抽帧
├── run_bon.py               # CLI入口（--mode bon / score）
├── Wan2.2/                  # 阿里视频生成模型（14B DiT, Flow Matching）
├── Depth-Anything-3-main/   # ByteDance深度估计模型（DA3）
├── WMReward-main/           # Meta参考实现（VJEPA-2 reward）
├── VGGTomega/               # 备选3D模型（未使用）
└── requirements.txt
```

## 工作流

- 本地（Windows）编辑代码 → git push → 远程服务器 git pull → 运行
- 远程服务器有GPU，本地不跑模型

## 运行命令

```bash
# 安装
pip install -r requirements.txt
pip install -e Depth-Anything-3-main/
export PYTHONPATH=$PYTHONPATH:$(pwd)/Wan2.2

# Best-of-N
python run_bon.py \
  --ckpt_dir /path/to/Wan2.2-I2V-A14B \
  --image /path/to/first_frame.png \
  --prompt "动作指令" \
  --N 8 --size 480*832 --sample_shift 3.0 --t5_cpu
```

## 关键技术细节

- DA3默认模型：`depth-anything/DA3NESTED-GIANT-LARGE-1.1`（1.4B）
- DA3 `inference()` 有 `@torch.inference_mode()`，Part 1 直接用；Part 2 需绕过它调用 `model.model`（DepthAnything3Net）
- DA3 输出：depth `(N,H,W)`, extrinsics `(N,4,4)` world-to-cam, intrinsics `(N,3,3)`, conf `(N,H,W)`
- Wan2.2 输出：`(3, T, H, W)` 值域 `[-1, 1]`，81帧，VAE时间压缩4x空间8x8
- Wan2.2 使用双DiT（high_noise_model + low_noise_model），boundary=0.9
- Flow Matching Tweedie估计：`x0 = x_t - sigma_t * velocity`
- Reward三分量：投射一致性50% + 首帧锚定35% + DA3置信度15%

## 开发计划

- **Part 1**（当前）：DA3 GeoReward + Best-of-N，不需要梯度
- **Part 2**（后续）：可微DA3 + 梯度引导去噪 + 与BoN组合

## 代码规范

- 用中文交流，代码和文件名用英文
- 不做多余抽象，保持代码直接可读
