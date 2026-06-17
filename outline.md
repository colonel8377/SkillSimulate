# Paper Outline: Digital Habitus Distillation

## Metadata
- **Title**: Digital Habitus Distillation: Beyond Roleplay in LLM-based Social Simulation
- **Target Venue**: ICWSM / WWW (10-12 pages)
- **Format**: English, LaTeX, ACM double-column (WWW/`acmart` template)
- **Core Narrative**: Sim-to-Real Gap in LLM social simulation → Root cause: Persona Prompting ceiling → CADP data-driven cognitive distillation → Prove better than roleplay (Exp 1) → Simulate real group dynamics (Exp 2)
- **Theory Positioning**: Bourdieu's Habitus serves as design inspiration (not as a tested theoretical claim). The three-dimensional mapping is used to motivate design choices; we do not claim to measure "habitus scores" or empirically validate the sociological framework itself.

---

## Chapter 1: Introduction (~1 page)
### 1.1 Background: LLM-based Social Simulation
- 社会科学家/政策制定者转向 LLM agent 模拟预测群体行为
- 代表性工作概述（Argyle et al. 2023; Park et al. 2023; Horton 2023）

### 1.2 The Sim-to-Real Gap
- LLM agents 产出交互系统性偏离真实人类动态
- 表现：行为同质化、冲突缺失、极化无法涌现、说服失败
- 引用 Wu et al. (2025) 的问题诊断 + "The Chameleon's Limit" (arXiv:2604.24698) 的 persona collapse 独立实证证据
- 引用 "What Limits LLM-based Human Simulation: LLMs or Our Design?" (arXiv:2501.08579) — 该论文质疑 sim-to-real gap 的根因是 LLM 能力限制还是设计缺陷；本文从设计角度（persona prompting ceiling）回应

### 1.3 Root Cause: The Ceiling of Persona Prompting
- Descriptive persona 只传递身份标签而非行为规则
- "Average Persona Problem" (Li & Cheng 2026)
- Persona collapse 根因：RLHF 压缩行为多样性 + 描述性 prompt 缺乏行为约束 ("The Chameleon's Limit")
- 引用 Cognitive Heuristics 论证 (Zhu & Heydari 2026)

### 1.4 Our Approach: CADP
- 三维蒸馏：Expression DNA / Mind Models / Anti-patterns
- 基于 nuwa-skill 框架编译
- 硬约束挂载

### 1.5 Contributions
1. 方法贡献: CADP 管线（dual-track compilation + three-tier enforcement）
2. 实证贡献 (Exp 1): 9 baselines × 3 datasets × 4 models 系统对比
3. 发现贡献 (Exp 2): 三真实社区的涌现动态

> Figure 1: 概览图

---

## Chapter 2: Related Work (~1.5 pages)
### 2.1 LLM-based Social Simulation
- Horton (2023), Park et al. (2023), Argyle et al. (2023)
- Park et al. (2024, Nature) — 大规模模拟 1,052 真实个体（interview-based persona），验证 generative agent 可信度；与 CADP 互补（interview-based vs behavioral-trace-based distillation）
- Gap: 无人系统验证 persona vs 数据驱动方法的行为保真度

### 2.2 The Homogenization Problem
- Wu et al. (2025) — 诊断平均人格问题，无解决方案
- Li & Cheng (2026) — audience segmentation，top-down 描述性方法
- "The Chameleon's Limit" (arXiv:2604.24698) — 独立实证证据：persona collapse（agent 行为收敛到 modal pattern）是结构性失效模式
- "What Limits LLM-based Human Simulation: LLMs or Our Design?" (arXiv:2501.08579) — 系统拆解 LLM 模拟偏差来源（模型能力 vs prompting 设计），结论指向设计缺陷为主因；本文进一步将"设计缺陷"定位到 persona prompting 的结构性局限并提出 CADP 作为改进方案
- "LLM-Based Social Simulations Require a Boundary" (arXiv:2506.19806) — 提出 LLM 社会模拟的有效性边界问题；本文在 §7.4 Threats to Validity 中讨论 CADP 的适用边界
- Gap: segmentation 不保证个体级行为保真度；persona collapse 的根因（RLHF 压缩 + 描述性 prompt 缺乏行为规则）未被解决；现有 gap 诊断缺乏可操作的结构性解决方案

### 2.3 Population-Aligned Persona Generation（新增 — 最直接竞争者）
- Population-Aligned Persona Generation (arXiv:2509.10127) — 用真实调查数据生成匹配人口**属性分布**的 persona 群体
- **关键区别**：该方法回答"有没有对的**类型**的人"（分布匹配）；CADP 回答"这些人**做**得对不对"（行为规则匹配）
- 该方法仍为描述性（人口学+态度属性），无行为规则传递，无硬约束，无群体交互模拟
- Gap: 分布匹配 ≠ 行为保真度；正确的人口学构成不保证正确的交互模式

### 2.3.5 Sociological Framing of AI Behavior（新增）
- "The taste, class and habitus of generative AI chatbots" (Sage Journals, 2025) — 将 Bourdieu 的 habitus 概念应用于分析 LLM 的 taste/class 表现
- **与本文的关系**：该论文在分析层面使用 habitus 概念（解读 AI 输出中的阶层偏好）；本文在方法设计层面借鉴 habitus 的三维结构（作为 distillation dimensions 的灵感来源），但**不声称验证 Bourdieu 理论本身**。两篇论文的 habitus 使用层次不同：analytic lens vs design inspiration

### 2.4 Behavioral Distillation & Cognitive Cloning
- COLLEAGUE.SKILL (Zhou et al. 2026) — 面向个人工具，不声称行为保真度；single-track compilation
- nuwa-skill 框架
- Zhu & Heydari (2026) — 理论推导，无实证
- Gap: 无工作将行为蒸馏应用于社会模拟并验证群体级保真度

### 2.5 Positioning Summary Table

| 维度 | Descriptive Persona | Segmentation | Pop-Aligned (2509.10127) | COLLEAGUE.SKILL | CADP (Ours) |
|------|--------------------|-------------|--------------------------|-----------------|-------------|
| 数据来源 | 人工/调查 | 人口学数据 | 调查数据 | 个人行为轨迹 | 社区行为轨迹 |
| 传递内容 | 身份标签 | 人口学+心理标签 | 人口分布属性 | 能力+部分行为边界 | 三维行为规则 |
| 挂载方式 | system prompt | system prompt | system prompt | .skill 文件 | .skill 硬约束 |
| 模拟目标 | 个体/群体 | 群体分布 | 群体分布匹配 | 个人工具 | 群体交互动态 |
| 约束机制 | 无 | 无 | 无 | 软建议 + negative examples | 三层强制执行 (per-dim α) |
| RLHF override | ✗ | ✗ | ✗ | 部分（negative examples 可部分抵消） | ✓ (Anti-patterns 硬约束) |
| 保真度验证 | 无 | 分布级 | 分布级 | 无 | 五层级系统验证 |

> **Table 说明**：✗/✓/部分 的区分旨在反映 continuum 而非 binary。COLLEAGUE.SKILL 中若包含 negative examples 则具有部分行为约束能力，但其约束为 advisory（无 enforcement mechanism），与 CADP 的三层硬约束在执行层面有结构性差异。

---

## Chapter 3: Theoretical Framework (~1 page)
### 3.1 Design Inspiration from Bourdieu's Habitus
- Bourdieu's Habitus 三维结构（语言性情 / 认知图式 / 禁忌）作为 CADP 三维设计（Expression DNA / Mind Models / Anti-patterns）的**灵感来源**
- **声明**：本文不声称测量或验证 habitus 本身。habitus 框架仅用于 motivate 三维设计的合理性和互斥性；评估（§5）直接测量行为保真度，不涉及 habitus 的社会学构念
- 与 "The taste, class and habitus of generative AI chatbots" (Sage, 2025) 的区别：该论文用 habitus 作为分析透镜解读 AI 行为；本文用 habitus 三维结构作为方法设计蓝图

### 3.2 Anti-patterns as Frame Selectors
- Goffman's Frame Analysis — frames 组织行为预期
- Anti-patterns = 框架切换触发器，克服 RLHF 妥协倾向
- **Persona collapse 的机制分析**：RLHF 通过偏好优化压缩输出分布（"Alignment Tax"），使 agent 在冲突场景中趋向妥协/回避。描述性 persona 无法对抗这一倾向，因为身份标签不影响生成时的 token 分布偏移。
- Anti-patterns 作为硬约束，通过 pre-generation block（Tier 3）和 post-generation filter（Tier 1）直接干预 token 级行为分布，结构性绕过 RLHF 的妥协压缩
- "The Chameleon's Limit" 的实证发现为这一机制分析提供数据支撑：persona collapse 的速率与 RLHF 强度正相关，验证了 anti-patterns 干预的必要性

---

## Chapter 4: Method — CADP Pipeline (~2 pages)
### 4.1 Overview
五步管线：Raw Corpus → Clustering → nuwa-skill Compilation → Agent Configuration → Sandbox Runtime

**方法学循环性防御声明**：聚类、编译、验证均基于同一社区数据，存在循环性质疑风险。本文通过四重机制防御：(1) 置换检验（§5.2 Condition 6 Shuffled）— 检测随机性；(2) 跨数据集迁移（§5.5 Wikipedia→Reddit）— 检测过拟合到特定社区；(3) Held-out 事件预测准确率（§5.3 Predictive Fidelity layer）— 检测方法是否捕获因果模式而非 spurious correlation；(4) 零样本跨社区迁移（§5.5 新增）— 在 community A 编译 .skill，直接应用于结构不同的 community B（如 Reddit → GitHub），若 fidelity 保持则说明捕获的是可泛化行为模式。Held-out 事件定义标准：由 2 名标注者独立编码争议性事件（conflict escalation / persuasion success / consensus formation），Cohen's κ ≥ 0.7 后取共识标签作为 ground truth。

### 4.2 Step 1: Clustering Typical Individuals
- 输入：平台完整对话日志
- 两阶段特征提取：
  - Stage 1: 行为信号 (reply depth, edit frequency, stance shift rate, conflict engagement ratio)
  - Stage 2: 语言嵌入 (sentence-BERT 或 domain-adapted embeddings)
  - 自适应权重拼接
- 聚类：K-Means / HDBSCAN, K=3-5
- K 选择标准：silhouette score + Davies-Bouldin index + 领域可解释性
- **Cluster stability**: 重采样 ARI variance < 0.2

### 4.3 Step 2: nuwa-skill Compilation (Dual-Track)
- Top-N 最具代表性交互（N≈20 对话线程）
- 输出**双轨 .skill 文件** (与 COLLEAGUE.SKILL 的关键结构性区别)：
  - COLLEAGUE.SKILL 是 single-track（仅 capability）；CADP 新增 Constraint Track，为硬约束操作化提供结构基础
  - **Capability Track** (agent 能做什么):
    - Expression DNA: 允许的语言模式、词汇范围、句法倾向
    - Mind Models: 推理模板、立场框架、评估标准
  - **Constraint Track** (agent 不能做什么):
    - Anti-patterns: 显式禁止列表 + 触发条件
    - 可强制执行的硬约束（COLLEAGUE.SKILL 的约束仅为软建议）
- **Compilation 流程区别**：
  - COLLEAGUE.SKILL: few-shot pattern extraction（ground-truth traces → capability rules），single-pass
  - CADP: dual-pass compilation — Pass 1 (positive case mining → capability rules) + Pass 2 (negative case mining → anti-pattern rules with trigger conditions)
  - Pass 2 的 anti-pattern detection prompt 工程：从低 fidelity 交互中提取"该 agent 不应做什么"，为每个 anti-pattern 生成结构化触发条件（见 §4.4 Trigger Formalization）

### 4.4 Step 3: Three-Tier Enforcement Mechanism (硬约束操作化)
.skill 作为硬约束，定义三层可度量执行机制：

| Tier | 维度 | 执行方式 | 时机 |
|------|------|---------|------|
| Tier 1 | Expression DNA | Post-generation embedding filter (2σ boundary reject + regenerate) | 生成后 |
| Tier 2 | Mind Models | Pre-generation context injection (rule conditioning) | 生成前 |
| Tier 3 | Anti-patterns | Pre-generation hard block (trigger match → forced reformulation) | 生成前 |

#### 4.4.1 Trigger Formalization（新增 — Anti-pattern 执行的形式化）
每个 anti-pattern 在编译阶段生成结构化触发器：
- **触发器表示**：三类混合匹配
  - 类别 A（词法级）：正则模式（如攻击性词汇、特定论辩结构标记）
  - 类别 B（语义级）：Sentence-BERT embedding cosine similarity ≥ θ_sem（θ_sem 通过 held-out 数据校准，默认 0.85）
  - 类别 C（行为级）：分类器（轻量 logistic regression over 行为特征向量：stance shift rate, conflict engagement ratio 等）
- **匹配逻辑**：任一类别触发即标记为 violation；触发阈值通过 held-out 交互数据上的 F1 优化校准
- **Trigger Calibration Protocol（独立校准实验 — §5 之外）**：
  - 标注协议：3 名标注者独立标注 500 条交互（per dataset），violation / non-violation 二分类
  - Inter-rater reliability：Fleiss' κ ≥ 0.6 后取多数票标签
  - 校准 / 验证分割：标注数据的 60% 用于 classifier 训练与阈值优化，40% 用于验证
  - 报告指标：Precision / Recall / F1（per trigger category A/B/C），而非仅 FP/FN rate
  - **目标性能**：Precision ≥ 0.90（避免误伤正常行为），Recall ≥ 0.80（捕获大部分违规）
  - 跨域迁移测试：在 Wikipedia 校准的 trigger classifier 直接应用于 Reddit，报告迁移后的 Precision/Recall 下降程度

#### 4.4.2 Forced Reformulation Protocol（新增）
当 Tier 3 触发器命中时：
1. **Block**：拒绝当前输出
2. **Diagnosis injection**：向 context 注入违规诊断（"你的上一条回复触发了 Anti-pattern: [pattern name]，因为 [trigger type] 匹配了 [rule]"）
3. **Constrained regeneration**：在增强 context 下重新生成（最多 N_retry=3 次）
4. **Fallback**：若 N_retry 后仍触发，降级为安全模板响应（标记为 constraint-forced，计入 evaluation）

#### 4.4.3 Constraint Hardness Parameter α（Per-Dimension Design）
- **α 为 per-dimension 向量** (α_expr, α_mind, α_anti)，分别控制三个维度的执行强度：
  - α=0: 对应维度退化为 advisory only（system prompt 中列出规则但不执行）
  - α=0.5: 对应 tier 概率执行（Tier 1: P(reject) ∝ embedding 距离 / 2σ；Tier 2: 50% 概率注入；Tier 3: P(block) ∝ trigger confidence）
  - α=1.0: 对应 tier 总是执行（Tier 1: 2σ reject；Tier 2: 总是注入；Tier 3: 总是 block）
- **Per-dimension 设计的理由**：不同社区类型中各维度的重要性不同（如 GitHub 的 Anti-patterns 侧重技术礼仪，Reddit 侧重论辩规则），单一全局 α 无法表达这种不对称
- **Tier 3 离散性说明**：α_anti=0.5 时，Tier 3 的 "概率 block" 本质上是 stochastic gating——以 trigger confidence 为概率决定是否 block。这是连续 α 与离散 hard block 之间的合理桥接
- **α Sensitivity Analysis（新增到 §5）**：扫描 α ∈ {0, 0.25, 0.5, 0.75, 1.0}^3（关键 cell），报告 sim-to-real gap 指标随 α 的变化曲线，确认最优 α 配置并验证 robustness
- **与消融实验的关系**：消融条件（minus Expression DNA / minus Mind Models / minus Anti-patterns）= 对应维度 α=0 且移除规则内容；α 调参是保留规则内容但调节执行强度。两者正交。

#### 4.4.4 模型适配策略（修订）
- **所有模型统一基线执行方案**：output filtering + re-prompting（Tier 1 embedding filter + Tier 3 forced reformulation via diagnosis injection）
- **开源模型增强（探索性）**：logit bias intervention 作为可选增强
  - 机制：对 anti-pattern token sequence 施加负 logit bias（bias 值通过 calibration set 回归确定）
  - **与现有工作的关系**：现有 constrained decoding 文献（JSON schema enforcement, grammar-guided decoding, safety refusal-and-retry 机制）处理的是**输出格式约束**或**安全过滤**。CADP 的 logit intervention 将约束对象从格式/安全扩展到**行为规则级别**（如论辩风格、交互模式约束），这一应用层面是新的，但底层机制（logit bias / constrained decoding）与现有 alignment 技术有结构性联系
  - 开源模型 logit intervention 定位为 **exploratory contribution**，非 main claim；若实验效果不显著则报告为 negative finding
- **API 模型 (GPT-4o, Claude)**：仅使用 output filtering + re-prompting（无法访问 logits）

### 4.5 Step 4: Agent Configuration & Population
- 按真实比例分配 N≈30 agents
- .skill 通过三层机制挂载为硬约束

### 4.6 Step 5: Constrained Interaction Sandbox
- 平台拓扑约束 + 平台特定动作空间：
  - Wikipedia: 编辑树结构; edit/revert/discuss/report
  - Reddit: threaded reply + delta; reply/award delta/counter-argue/block
  - GitHub: issue lifecycle; comment/label/close/reopen/assign
- 议题：从真实数据采样
- Agent runtime: memory (importance-weighted retrieval) + reflection + planning
  - Memory: 每轮对话历史
  - Reflection: 周期性信念巩固 (Mind Models reinforcement)
  - Planning: 每轮参考 .skill 约束选择动作
- 30-50 轮/条件，重复 5-10 次

### 4.7 Baselines (扩展为 9 条件)
1. Vanilla LLM
2. Descriptive Persona
3. Segmentation Persona (复现 Li & Cheng 2026)
4. **Population-Aligned Persona (复现 arXiv:2509.10127)** — 最直接竞争者，属性分布匹配
5. CADP (Full)
6. CADP (Shuffled) — 置换检验
7. **CADP minus Expression DNA** — 消融
8. **CADP minus Mind Models** — 消融
9. **CADP minus Anti-patterns** — 消融

> Figure 2: CADP Pipeline 流程图 (含 dual-track skill + three-tier enforcement)

---

## Chapter 5: Experiment 1 — Method Validation (~3 pages)
### 5.1 Setup
- 数据集: Wikipedia Talk Pages / Reddit r/changemyview / GitHub Issues
- 模型: GPT-4o / Claude 3.5 Sonnet / Llama-3-70B / Qwen-2.5-72B
- 总条件数: 9 × 3 × 4 = 108 (含 Pop-Aligned baseline)
- 重复: 每条件 5-10 次
- **样本量与可行性说明**：108 cells × 10 repeats = 1,080 simulation runs per metric；API 成本估算（GPT-4o + Claude 3.5）约 $2,000–5,000；开源模型推理（Llama-3-70B, Qwen-2.5-72B）需 4× A100 80GB，估算 200–400 GPU-hours
- **Power analysis**：基于 pilot data（Wikipedia 单数据集）的 effect size 估计，Cohen's d ≥ 0.5 时 5 次重复即可达到 80% power (α=0.05)；若 pilot 显示 d < 0.5，则增加至 10 次重复

### 5.2 Baselines (扩展为 9 条件)
1. Vanilla LLM (无 persona)
2. Descriptive Persona (标准 system prompt)
3. Segmentation Persona (复现 Li & Cheng 2026)
4. **Population-Aligned Persona (复现 arXiv:2509.10127)** — 最直接竞争者，测试"属性分布匹配"能否达到"行为规则蒸馏"的保真度
5. CADP (Full) — 三维完整
6. CADP (Shuffled) — 置换检验
7. **CADP minus Expression DNA** — 去语言维度消融
8. **CADP minus Mind Models** — 去认知维度消融
9. **CADP minus Anti-patterns** — 去反模式消融 (测 RLHF override 假设)

> 消融实验目的：隔离三维各自贡献，回应"维度选择是否arbitrary"的审稿质疑
> 预期：Anti-patterns 移除对冲突/极化指标影响最大；Expression DNA 移除对语言指标影响最大
> 关键对比预期：CADP (Full) vs Pop-Aligned — 属性分布匹配能接近但无法达到行为规则级的保真度，差距在 Micro Behavior 和 Predictive Fidelity 层最显著

### 5.3 Five-Layer Evaluation Metrics
- **Macro Topology**: ΔQ Modularity, E-I Polarization Index, NED, **Coverage** (行为空间覆盖率, from Xiao et al. 2026)
  - Ground truth: 从真实社区同期交互日志中提取的网络结构（同一时间窗口的交互图）
- **Meso Dynamics**: Cascade Length Fit (KS-test), DTW, **Structural Fidelity** (交互网络结构相关, from Li & Cheng 2026)
  - Ground truth: 真实社区中已发生争议事件的 cascade 长度分布、时间序列
- **Micro Behavior**: Action Matrix Similarity (Frobenius), RSA, **Uniformity** (行为分布熵, 检测同质化), **Complexity** (跨agent行为方差)
  - Ground truth: 按聚类类型匹配的真实用户子群的行为分布（simulated cluster i ↔ real cluster i 的行为向量）
- **Linguistics**: LSM (KL-divergence), SIP (Sentence-BERT cosine)
  - ⚠️ **Feature Leakage 注意**：Expression DNA 的蒸馏特征包含用词分布、句法模式。为避免自证循环，Linguistics 层评估使用**独立特征空间**——采用 Expression DNA 蒸馏时未使用的 NLP 特征（如 discourse marker 分布、sentiment trajectory shape、speech act ratio）作为评估特征，与蒸馏特征空间正交
- **Predictive Fidelity** (新增第五层, from Li & Cheng 2026):
  - 用仿真预测 held-out 真实交互结果 (谁会冲突、说服是否成功、冲突是否升级)
  - Ground truth 编码协议：2 名标注者独立编码 held-out 事件结果，Cohen's κ ≥ 0.7 后取共识标签
  - 这是最强的 "so what" 指标 — 仿真能否预测未见过的事件

### 5.3.5 Trigger Calibration Experiment (新增 — §4.4.1 的独立验证)
- **目的**：独立评估 anti-pattern trigger classifier 的检测性能，作为 Three-Tier Enforcement 的前提验证
- **数据**：per dataset 500 条标注交互（3 名标注者，Fleiss' κ ≥ 0.6）
- **Split**：60% train / 40% test
- **报告**：
  - Per trigger category (A 词法 / B 语义 / C 行为) 的 Precision / Recall / F1
  - 阈值 sensitivity（θ_sem ∈ {0.75, 0.80, 0.85, 0.90, 0.95}）
  - 跨数据集迁移性能（Wikipedia-trained → Reddit test）
- **通过标准**：Precision ≥ 0.90, Recall ≥ 0.80（否则报告为 limitation 并讨论对主实验的影响）

### 5.4 Cluster Stability Validation (新增)
- 多次重采样后计算 Adjusted Rand Index (ARI)
- ARI variance < 0.2 视为聚类稳定
- 配合 silhouette score + Davies-Bouldin index 选择 K
- 目的：证明聚类发现的行为类型是 robust pattern，非 artifact

### 5.5 Cross-Dataset Transfer Test (新增)
- **同类型跨数据集迁移**：在 Wikipedia 上编译 .skill → 直接应用于 Reddit（均为在线异步文本讨论）
- **跨结构零样本迁移（新增）**：在 Reddit r/changemyview 上编译 .skill → 直接应用于 GitHub Issues（交互拓扑和动作空间完全不同）
  - 若跨结构迁移仍保持部分 fidelity → 证明 CADP 捕获的是可泛化行为模式（超越了平台特定 artifact）
  - 这是对方法学循环性的最强防御：不同交互结构的社区间迁移排除了"过拟合到平台特征"的解释
- 报告：迁移后 vs 原位编译的 fidelity 差距，以及 trigger classifier 的迁移性能下降程度
- 直接回应方法学循环性质疑

### 5.6 Human Evaluation (新增)
- 3 名领域专家盲评 100 条仿真对话
- 条件盲分配 (CADP vs Descriptive Persona vs Real)
- Cohen's κ ≥ 0.6 为可接受评分一致性
- 目的：外部效度补充，防止 metric overfitting

### 5.6.5 α Sensitivity Analysis (新增)
- 扫描 per-dimension α ∈ {0, 0.25, 0.5, 0.75, 1.0}，重点报告：
  - α_expr vs α_anti 的独立影响曲线（Mind Models 固定为 α=1.0）
  - 不同社区类型（Wikipedia / Reddit / GitHub）的最优 α 配置差异
  - 确认 CADP 性能对 α 选择的 robustness（最优区域是否 plateau 而非尖锐 peak）
- 目的：证明 per-dimension α 的必要性（不同社区需要不同配置）并验证 enforcement 强度的可控性

### 5.7 Results
- CADP (Full) 全面显著优于所有 baseline (9 条件对比)
- 消融分析: 各维度独立贡献量化
- 关键对比: CADP vs Segmentation Persona 逐维度差异；CADP vs Population-Aligned Persona 的行为保真度差异
- 置换检验: Shuffled 显著弱于 Full
- 跨数据集迁移: Wikipedia→Reddit fidelity 保持程度；Reddit→GitHub 跨结构零样本迁移结果
- 预测性保真度: held-out 事件预测准确率
- Trigger calibration: per-category P/R/F1 + 跨域迁移性能
- α Sensitivity: per-dimension α 曲线 + 不同社区最优配置
- 跨模型/跨数据集一致性
- **Human evaluation 结果**：3 名专家盲评（Cohen's κ ≥ 0.6）中 CADP 的辨识度——专家能否区分 CADP 仿真 vs 真实交互（作为外部效度的核心证据，在主结果中报告）

> Table 1: 主结果表 (9 baselines × 5 metric layers)
> Table 2: 消融结果表 (3 维度 × 5 metric layers)
> Figure 3: 雷达图 — 9 条件 5 层指标对比
> Figure 4: 交互网络可视化对比

### 5.8 Analysis
- 各维度贡献分析 (基于消融)
- Descriptive Persona 在哪些指标上最接近 CADP
- **CADP vs Pop-Aligned 深入对比**：在哪些 metric layer 差距最大/最小？Pop-Aligned 在 Macro Topology（分布级）可能接近，但在 Micro Behavior 和 Predictive Fidelity（行为级）预期显著落后——量化"属性匹配 ≠ 行为匹配"
- **回应 "What Limits LLM Simulation" (arXiv:2501.08579)**：跨模型对比分析——若 CADP 在不同模型上的提升幅度一致，说明 design fix（CADP）的效果不依赖模型能力；若在更强模型上提升幅度递减，则支持"LLMs and Design 共同限制"的交互效应假设
- Anti-patterns 作为 RLHF override 的作用机制
- α Sensitivity: per-dimension 最优配置的社区差异性
- Predictive fidelity 的 "so what" 论证

---

## Chapter 6: Experiment 2 — Social Simulation (~2.5 pages)
### 6.1 Motivation
Exp 1 证明 CADP 可信 → Exp 2 模拟真实社区

### 6.2 Setup
- 从真实数据采样争议性场景
- CADP agents 交互 50 轮
- 对比：真实历史 vs CADP vs Descriptive Persona
- **Scale test**: 主实验 N≈30，附加 N≈100 条件测试 fidelity 随规模保持度

### 6.3 Platform-Specific Action Space Enforcement
每个平台强制执行真实交互拓扑和动作空间：
- **Wikipedia**: 编辑树结构 (edit → revert → re-edit)；动作空间: edit / revert / discuss / report
- **Reddit r/changemyview**: 线程回复 + delta 机制；动作空间: reply / award delta / counter-argue / block
- **GitHub Issues**: Issue 生命周期 (open → comment → close/reopen)；动作空间: comment / label / close / reopen / assign

### 6.4 Temporal Trajectory Analysis (新增)
- 不仅对比终态，还对比**演化轨迹**
- 极化指数按轮次画演化曲线，DTW 对齐真实时间线
- 追踪关键事件时机：首次冲突、首次说服成功的出现轮次 vs 真实时间点
- 目的：终态匹配 ≠ 过程匹配，轨迹对比更严格

### 6.5 Results by Dataset
- **Wikipedia**: 编辑冲突动态，极化指数演化曲线，编辑战 escalation pattern
- **Reddit**: 说服成功率 (delta award rate)，对抗网络结构，反驳链深度
- **GitHub**: 技术共识形成路径，群体分化时间线

### 6.6 Cross-Dataset Comparison
- 极化动态对比 (轨迹层面)
- CADP vs Descriptive Persona 能"看见"什么 vs "看不见"什么
- Scale test: N=30 vs N=100 fidelity 变化

> Figure 5: 极化指数时间演化曲线 (真实 vs CADP vs Descriptive Persona, 三平台)
> Figure 6: 模拟交互网络快照 (多时间点)
> Figure 7: 关键事件时机对比 (CADP vs real vs baseline)

### 6.7 Exploratory Findings
- 非预期的群体动态现象
- 在真实数据中验证对应

---

## Chapter 7: Discussion (~1 page)
### 7.1 Why CADP Outperforms Persona: The Alignment Tax
- RLHF 压缩行为多样性
- Anti-patterns 作为硬约束覆盖 RLHF 妥协倾向
- 三维协同必要性

### 7.2 Implications for Social Simulation Methodology
- 从"描述人是谁"到"蒸馏人怎么做"
- 对 ICWSM 社区的方法论建议

### 7.3 Relationship to Competing Methods
- vs COLLEAGUE.SKILL: 不同目标（个人工具 vs 群体模拟）+ 结构性区别（single-track vs dual-track + constraint enforcement）
- vs Population-Aligned Persona Generation: 不同层级（属性分布匹配 vs 行为规则蒸馏）；非替代关系，可叠加
- vs Restoring Heterogeneity (Wu et al. 2025): 补充而非替代（诊断 vs 解决方案）
- vs Cognitive Heuristics (Zhu & Heydari 2026): 实证验证（理论推导 vs 可操作 pipeline）
- vs The Chameleon's Limit: 共同问题诊断，CADP 提供结构性解决方案
- vs "What Limits LLM Simulation" (arXiv:2501.08579): CADP 是该论文提出问题（LLMs or Design?）的 design-side 回答；跨模型分析（§5.8）量化 design fix 相对于 model capability 的边际贡献
- vs Park et al. (2024, Nature, 1,052 individuals): interview-based vs behavioral-trace-based distillation；interview 依赖主动参与，CADP 可大规模从被动数据蒸馏
- vs "Habitus of GenAI" (Sage, 2025): 分析层面 vs 方法设计层面使用 habitus 概念

### 7.4 Threats to Validity（扩展为 Threats to Validity）
- **方法学循环性**：聚类→编译→验证基于同一社区数据。四重防御：(1) 置换检验，(2) 跨数据集迁移 (Wikipedia→Reddit, 同类型)，(3) held-out 事件预测（带标注者间一致性编码），(4) 跨结构零样本迁移（Reddit→GitHub，交互拓扑完全不同）。四者一致指向 CADP 优势时循环性影响可控。
- **与 "What Limits LLM Simulation" (arXiv:2501.08579) 的关系**：该论文将 sim-to-real gap 归因为"LLMs 或 Our Design"。本文从 design 角度回应（persona prompting ceiling），但未排除 LLM 能力限制的交互效应——若更强大模型自然缩小 gap，CADP 的边际贡献可能随模型规模递减。本文通过 4 模型对比部分控制此威胁，但未来模型的能力提升可能改变结论。
- **与 LLM Simulation Boundary (arXiv:2506.19806) 的关系**：该论文提出 LLM 模拟的有效性边界。CADP 的适用边界：当社区行为高度依赖平台外知识（如线下社会关系、跨平台历史）时，仅从单平台行为轨迹蒸馏的 .skill 可能不足。
- **与 Population-Aligned Persona 的区分边界**：Pop-Aligned (arXiv:2509.10127) 做属性分布匹配，CADP 做行为规则蒸馏。两者非替代关系——理论上可叠加（Pop-Aligned 选人 + CADP 赋予行为），但本文实验未测试叠加效果。
- **Logit steering 可行性限制**：行为规则级 logit intervention 与现有 constrained decoding 有结构性联系，应用层面（行为规则 vs 格式/安全）是新的。开源模型增强方案效果不确定，定位为 exploratory contribution。
- **Anti-pattern trigger 校准依赖**：trigger 阈值需 per-dataset 校准（§4.4.1），跨域迁移时 Precision/Recall 会下降——§5.5 报告具体下降程度。
- **Anti-patterns 编码社区偏见的风险**：CADP 从真实社区行为中蒸馏 anti-patterns，可能忠实地再现社区中的偏见性规范（如隐性歧视行为模式）。这是 "fidelity vs. ethics" 的结构性张力。本文在 §7.5 Ethics 中讨论缓解策略，但承认无法完全消除。
- **Bourdieu 框架的可证伪性**：本文将 habitus 作为 design inspiration 而非 tested theory，因此不涉及可证伪性风险。但读者可能期待 habitus → CADP 三维映射的实证验证——本文仅提供间接证据（三维消融各自有显著贡献），不做直接的 sociological construct validation。
- **模型依赖**：4 模型验证覆盖主流 API + 开源，但未测试 smaller models (<7B)。
- **平台覆盖**：仅英文平台，跨语言行为模式差异未验证。
- **时间窗口**：训练数据为特定时间段的交互快照，行为模式的时间漂移未建模。

### 7.5 Ethical Considerations（扩展）
- **IRB 与数据合规**：所有数据来自公开平台 API（CC-BY-SA / 平台 ToS 允许研究用途）；已通过 IRB 审查（或豁免）
- **PII 处理**：用户名、IP 引用、个人链接等 PII 在预处理阶段移除
- **Anti-patterns 与社区偏见**（新增）：CADP 的核心方法——从真实社区行为中蒸馏 anti-patterns——意味着如果社区存在系统性偏见（如性别刻板印象、种族偏见行为模式），CADP 会忠实地编码这些偏见。这是 "behavioral fidelity" 的直接后果：
  - 缓解：在编译阶段增加 bias audit step——检测 anti-patterns 中是否包含 protected class 相关的行为规则；若检测到，标记并人工审核
  - 限制：bias audit 无法覆盖所有隐性偏见；报告为 acknowledged limitation
- **Dual-use 风险**（新增）：CADP 的行为蒸馏能力可用于恶意目的（如精确模拟特定个体进行 social engineering）。缓解：不公开个体级 .skill 文件；仅发布聚合级模型和 pipeline 代码；添加 responsible use statement
- **模拟个体的同意问题**（新增）：被模拟的用户未明确同意被"数字克隆"。当前合规基础为公开数据的研究豁免，但长期来看需要社区治理框架

---

## Chapter 8: Conclusion (~0.5 page)
- 重申问题、方案、发现
- 展望：反事实实验、跨平台迁移、Alignment Tax 量化

---

## Tables & Figures Summary

| # | Content | Section |
|---|---------|---------|
| Table 1 | 9 baselines × 5 metric layers main results | 5.7 |
| Table 2 | Ablation results (3 dimensions × 5 metric layers) | 5.7 |
| Table 3 | Positioning comparison (5 methods × 7 dims) | 2.5 |
| Table 4 | Three-tier enforcement mechanism + trigger formalization | 4.4 |
| Table 5 | Trigger calibration results (P/R/F1 per category × 3 datasets) | 5.3.5 |
| Table 6 | α Sensitivity: per-dimension α × metric layers (key cells) | 5.6.5 |
| Figure 1 | Overview diagram | 1.5 |
| Figure 2 | CADP Pipeline flowchart (dual-track skill + three-tier enforcement) | 4.1 |
| Figure 3 | Radar chart — 9 conditions × 5 metric layers | 5.7 |
| Figure 4 | Interaction network visualization comparison | 5.7 |
| Figure 5 | Polarization index time evolution curves (real vs CADP vs baseline, 3 platforms) | 6.5 |
| Figure 6 | Simulated interaction network snapshots (multi-timepoint) | 6.6 |
| Figure 7 | Key event timing comparison (CADP vs real vs baseline) | 6.6 |
| Figure 8 | α Sensitivity heatmaps (per-dimension α × 5 metric layers, per dataset) | 5.6.5 |
| Figure 9 | Cross-structure transfer fidelity (Reddit→GitHub) | 5.5 |
