# 📘 策略梯度（Policy Gradient）完整数学推导

## 1️⃣ 问题定义：最大化期望回报

我们有一个参数化策略：
$$
\pi_\theta(a|s) = \mathbb{P}(a_t = a | s_t = s; \theta)
$$
目标是找到最优参数 $\theta$，使得**期望折扣累积回报最大**：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
$$

其中：

- $$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$$ 是一条轨迹（trajectory）
- $s_0 \sim \rho_0(\cdot)$：初始状态分布
- $a_t \sim \pi_\theta(\cdot|s_t)$：策略采样动作
- $s_{t+1} \sim P(\cdot|s_t, a_t)$：环境转移（与 $\theta$ 无关）
- $\gamma \in [0,1)$：折扣因子

## 2️⃣ 核心目标：计算梯度 $\nabla_\theta J(\theta)$

直接对期望求导：

$$
\nabla_\theta J(\theta) = \nabla_\theta \int p(\tau; \theta) R(\tau) \, d\tau
$$

其中 $p(\tau; \theta)$ 是轨迹 $\tau$ 在策略 $\pi_\theta$ 下的概率密度：

$$
p(\tau; \theta) = \rho_0(s_0) \prod_{t=0}^{T} \pi_\theta(a_t|s_t) P(s_{t+1}|s_t, a_t)
$$

注意：只有 $\pi_\theta(a_t|s_t)$ 依赖 $\theta$，环境动力学 $P$ 和初始分布 $\rho_0$ 不依赖 $\theta$。

## 3️⃣ 关键技巧：Log-Derivative Trick（似然比方法）

我们使用恒等式：

$$
\nabla_\theta p(\tau; \theta) = p(\tau; \theta) \nabla_\theta \log p(\tau; \theta)
$$

代入梯度表达式：

$$
\nabla_\theta J(\theta) = \int \nabla_\theta p(\tau; \theta) R(\tau) \, d\tau = \int p(\tau; \theta) \nabla_\theta \log p(\tau; \theta) R(\tau) \, d\tau
$$

即：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log p(\tau; \theta) \cdot R(\tau) \right]
$$

## 4️⃣ 展开 $\log p(\tau; \theta)$

$$
\log p(\tau; \theta) = \log \rho_0(s_0) + \sum_{t=0}^{T} \left[ \log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t, a_t) \right]
$$

对 $\theta$ 求导，不含 $\theta$ 的项导数为 0：

$$
\nabla_\theta \log p(\tau; \theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)
$$

代入梯度：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \left( \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \right) \cdot R(\tau) \right]
$$

交换求和与期望：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R(\tau) \right]
$$

## 5️⃣ 重要重构：按时间步分配回报

注意到 $R(\tau)$ 是整条轨迹的回报，但我们可以将它“归属”到每个时间步：

定义从时间 $t$ 开始的折扣回报：

$$
R_t = \sum_{k=t}^{T} \gamma^{k-t} r_k
$$

于是：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t \right]
$$

（因为 $R(\tau)$ 中只有 $R_t$ 与 $(s_t, a_t)$ 之后的动作有关，前面的动作不影响 $R_t$）

## 6️⃣ 转换为状态-动作期望形式

我们可以将轨迹期望，写成在“折扣状态访问分布”下的期望：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta(\cdot|s)} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a) \right]
$$

其中：

- $d^{\pi_\theta}(s) = \sum_{t=0}^{\infty} \gamma^t \mathbb{P}(s_t = s | \pi_\theta)$ 是折扣状态分布
- $Q^{\pi_\theta}(s, a) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R_t | s_t = s, a_t = a \right]$ 是动作值函数

## 7️⃣ 引入优势函数（降低方差）

因为 $Q^{\pi_\theta}(s,a)$ 包含环境噪声，方差大，我们引入基线 $b(s)$：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{s,a} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot \left( Q^{\pi_\theta}(s,a) - b(s) \right) \right]
$$

只要 $b(s)$ 不依赖 $a$，就不会引入偏差。

最优选择是状态值函数：

$$
b(s) = V^{\pi_\theta}(s) = \mathbb{E}_{a \sim \pi_\theta} \left[ Q^{\pi_\theta}(s,a) \right]
$$

于是我们定义**优势函数**：

$$
A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)
$$

最终策略梯度公式：

$$
\boxed{
\nabla_\theta J(\theta) = \mathbb{E}_{s \sim d^{\pi_\theta}, a \sim \pi_\theta(\cdot|s)} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_\theta}(s,a) \right]
}
$$

## 8️⃣ 实际算法：REINFORCE with Baseline

因为我们不知道 $A^{\pi_\theta}$，实践中用采样估计：

1. 用 $\pi_\theta$ 采样一条完整轨迹：$\tau = \{ (s_t, a_t, r_t) \}_{t=0}^T$
2. 计算回报：$G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$
3. 估计优势：$\hat{A}_t = G_t - \hat{V}(s_t)$（$\hat{V}$ 可用另一个网络估计，或忽略）
4. 梯度估计：

$$
\hat{g} = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t
$$

5. 更新参数：

$$
\theta \leftarrow \theta + \alpha \hat{g}
$$

这就是著名的 **REINFORCE 算法**。

## 9️⃣ 示例：高斯策略（连续动作空间）

设策略为高斯分布：

$$
\pi_\theta(a|s) = \mathcal{N}(a; \mu_\theta(s), \sigma^2 I)
$$

则：

$$
\log \pi_\theta(a|s) = -\frac{1}{2} \log(2\pi\sigma^2) - \frac{(a - \mu_\theta(s))^2}{2\sigma^2}
$$

梯度：

$$
\nabla_\theta \log \pi_\theta(a|s) = \nabla_\theta \mu_\theta(s) \cdot \frac{a - \mu_\theta(s)}{\sigma^2}
$$

（如果 $\sigma$ 固定）

更新方向：如果动作 $a$ 比均值 $\mu$ 大，且优势为正 → 鼓励增大 $\mu$ → 下次更可能选大动作！

## 🔟 为什么 PG 不稳定？（为 TRPO/PPO 做铺垫）

虽然数学正确，但：

- 梯度是“一阶局部近似”，步长 $\alpha$ 难选
- 如果 $\theta$ 更新太大，采样分布 $d^{\pi_\theta}$ 和优势估计 $A^{\pi_\theta}$ 就不准了
- 可能“一步天堂，一步地狱” → 训练崩溃

👉 这就是为什么我们需要 **约束策略更新的幅度** —— 于是 TRPO 登场！

## 🧩 小明骑车的数学映射

- $s_t$：车身倾斜角度、速度
- $a_t$：把手转动角度
- $\pi_\theta(a|s)$：神经网络输出的“推荐动作分布”
- $A(s,a)$：这个动作比“平均动作”好多少
- $\nabla_\theta \log \pi$：如果这次右转5度稳住了车（$A>0$），就让网络下次在类似状态下更倾向输出“右转5度”

但如果一次更新让网络从“右转5度”变成“右转50度”，$A$ 的估计就失效了 → 翻车！

✅ 至此，Policy Gradient 的数学推导完成！

## 📎 附：常用符号表

| 符号 | 含义 |
|------|------|
| $\pi_\theta(a|s)$ | 策略函数，参数化概率分布 |
| $J(\theta)$ | 期望总回报目标函数 |
| $R_t$ | 从时间 $t$ 开始的折扣回报 |
| $A^{\pi}(s,a)$ | 优势函数 = $Q^\pi - V^\pi$ |
| $d^\pi(s)$ | 折扣状态访问分布 |
| $\nabla_\theta \log \pi$ | 策略梯度方向（似然比） |