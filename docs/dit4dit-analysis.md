# DiT4DiT 模型分析与 GAP Adapt 方案

> repo: https://github.com/xiaochy/DiT4DiT  
> 原始论文: arXiv:2603.10448 — *Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control*

---

## 1. 核心思想

DiT4DiT 是一个 **Vision-Action-Model (VAM)** 框架，核心想法是：

> 用**视频生成模型**（Cosmos-Predict2.5-2B）作为 backbone，把其内部 transformer 的隐藏特征（而不是最终生成的视频）抽出来，作为条件信号，驱动一个 **Flow Matching 动作头**预测机器人动作。

两部分（视频 FM backbone + 动作 FM head）可以联合训练（`training: joint`），也可以只训练动作头（`training: action`）。

---

## 2. 整体架构

```
观测帧序列 (T frames) + 语言指令
          │
          ▼
┌──────────────────────────────────┐
│    Cosmos-Predict2.5-2B          │  ← NVIDIA 视频扩散 transformer
│    (Video Diffusion Transformer) │
│    extract_layer = 17            │  ← 从第17层 hook 出隐藏特征
└──────────────┬───────────────────┘
               │ [B, seq_len, 2048]   VL 特征 (视觉+语言融合)
               ▼
┌──────────────────────────────────┐
│    FlowmatchingActionHead        │
│                                  │
│  noise ──► ActionEncoder ──────► │─── cross-attention ──► action tokens
│            (MLP + SinPE)         │    (attend to VL features)
│                                  │
│  ┌──────────────────────────┐    │
│  │  DiT-B  (16 layers)      │    │
│  │  AdaLN(timestep)         │    │
│  │  cross_attn_dim = 2048   │    │
│  └──────────────────────────┘    │
│                                  │
│  ActionDecoder (MLP)             │  → 预测速度场 v = ε - x₀
└──────────────────────────────────┘
               │
               ▼
    动作轨迹 [B, 16, action_dim]
    (Euler 积分 4 步, noise → clean)
```

---

## 3. Flow Matching 动作头细节

### 训练
- 干净动作 $x_0$，采样时间步 $t \sim \text{Beta}(1.5, 1.0)$
- 加噪：$x_t = (1-t)\,x_0 + t\,\epsilon$，目标速度场 $v = \epsilon - x_0$
- ActionEncoder 把 $(x_t,\, t)$ 编码为 token
- DiT 以 VL 特征为 cross-attn 条件，预测速度
- Loss：$(v_{\text{pred}} - v_{\text{target}})^2 \cdot \text{action\_mask}$

### 推理
- 从 $x_T = \epsilon$ 出发，Euler 积分 4 步（`num_inference_timesteps=4`）
- 每步 $x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t, \text{VL})$

### DiT 内部结构
每层 `BasicTransformerBlock` 结构：
```
AdaLN(timestep) → Cross-Attention(Q=action, KV=VL) → FFN
```
奇数层可选开 self-attention（`interleave_self_attention=True`），偶数层强制 cross-attn。

---

## 4. 关键文件

| 文件 | 作用 | 重点关注 |
|------|------|---------|
| `DiT4DiT/model/framework/DiT4DiT.py` | 主框架，forward / predict_action | action_mask 逻辑，repeated_diffusion_steps |
| `DiT4DiT/model/modules/action_model/ActionDiT.py` | FlowMatching 全部逻辑 | `forward()` 加噪流程，`predict_action()` Euler 积分 |
| `DiT4DiT/model/modules/action_model/flow_matching_head/cross_attention_dit.py` | DiT transformer 本体 | `BasicTransformerBlock`，`AdaLayerNorm` |
| `DiT4DiT/model/modules/action_model/flow_matching_head/action_encoder.py` | action + timestep 编码 | `ActionEncoder`，`SinusoidalPositionalEncoding` |
| `DiT4DiT/model/modules/vlm/Cosmos25.py` | Cosmos backbone + hook | `_register_hook()`，`build_cosmos_inputs()` |
| `DiT4DiT/config/robocasa/dit4dit_robocasa_gr1.yaml` | 完整超参配置 | cross_attention_dim, num_layers, action_horizon |

---

## 5. 与 GAP 的对比

| 维度 | GAP（当前） | DiT4DiT |
|------|------------|---------|
| 视觉 backbone | DINO + 自定义编码 | Cosmos-2.5 视频扩散 transformer |
| 动作头 | DiT（含 aux track loss） | DiT + Flow Matching（更稳定） |
| Track 预测 | 独立 auxiliary task，MSE loss | 无（可借鉴其 FM 框架来改造） |
| 条件信号 | 视觉 token | VL 融合 token（语言也进 cross-attn） |
| 推理步数 | 1 步（DDIM） or 多步 | 4 步 Euler |
| Joint 训练 | track loss + action loss | video FM loss + action FM loss |

---

## 6. Adapt 到 GAP 的方案

### 方案 A：把 Track Token 加入 cross-attention 条件（低风险，最先做）

Track prediction 的中间特征当前只流向 track loss head，没有反哺 action head。

```python
# 现在（GAP action head）:
action_output = dit(noisy_action, vl_features)

# 改造：把 track 特征拼到条件序列
track_tokens = track_encoder(cotracker_output)   # [B, N_track, D]
fused_cond   = torch.cat([vl_features, track_tokens], dim=1)  # seq 维拼接
action_output = dit(noisy_action, fused_cond)    # cross-attn 自然选择相关 token
```

**优点**：不改 DiT 结构，cross-attn 天然支持可变长度条件。  
**改动文件**：GAP 的 action head forward，加一个 `track_encoder` MLP。

---

### 方案 B：Track 作为 DiT 输入端的前缀 Token（中风险）

参考 DiT4DiT 的 `state_encoder`（MLP 把 state 编码后 concat 到 action token 前面）：

```python
# DiT4DiT 现有：
sa_embs = cat(state_features, action_features)   # [B, 1 + T_action, D]

# GAP 改造：
track_embs = track_mlp(track_pred)               # [B, N_track, D]
sa_embs = cat(track_embs, state_features, action_features)
# track token 通过 interleave self-attention 直接与 action token 交互
```

**优点**：track 信息在 self-attention 层与 action token 深度交互，不只是条件。

---

### 方案 C：Track 也换成 Flow Matching Head（最激进，最接近 DiT4DiT 精神）

当前 track loss 是 MSE。改成 FM 后：

```python
# 两个 FM head 共享同一个 VL backbone 特征
action_loss = action_fm_head(vl_features, action_targets)
track_loss  = track_fm_head(vl_features, track_targets)   # 复用 cross_attention_dit.DiT
total_loss  = action_loss + λ * track_loss
```

**优点**：track 预测引入扩散式不确定性建模，对遮挡/模糊更鲁棒；两个 head 结构对称，便于后期 distillation。  
**改动文件**：新增 `TrackDiTHead`（直接复用 `DiT4DiT/model/modules/action_model/flow_matching_head/cross_attention_dit.py` 的 `DiT` 类）。

---

## 7. 建议的推进顺序

```
第一步（1-2天）
  └── 方案 A：track token → fused_cond → action cross-attn
      验证 track 信息能否提升 action 精度

第二步（如果 A 有效）
  └── 方案 B：track 作为前缀 token，开 interleave self-attn
      让 action 和 track 在 token 层面直接交互

第三步（论文级别提升）
  └── 方案 C：track 换 FM head，joint 训练
      复用 cross_attention_dit.DiT，只改 input/output dim
```

---

## 8. 直接可复用的代码模块

| DiT4DiT 模块 | 可直接复用到 GAP 的场景 |
|-------------|----------------------|
| `cross_attention_dit.py::DiT` | Track FM head（改 action_dim → track_dim） |
| `ActionDiT.py::ActionEncoder` | Track token 编码（改 action_dim → track_dim） |
| `ActionDiT.py::FlowmatchingActionHead.forward()` | 加噪/速度场 loss 的完整实现 |
| `ActionDiT.py::FlowmatchingActionHead.predict_action()` | Euler 积分推理，直接 copy |
| `action_encoder.py::SinusoidalPositionalEncoding` | 时间步 embedding |
