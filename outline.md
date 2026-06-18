# Paper Outline: Digital Habitus Distillation

## Metadata
- **Title**: Digital Habitus Distillation: Beyond Roleplay in LLM-based Social Simulation
- **Target Venue**: ICWSM / WWW (10-12 pages)
- **Format**: English, LaTeX, ACM double-column (WWW/`acmart` template)
- **Core Narrative**: Sim-to-Real Gap in LLM social simulation → Root cause: Persona Prompting ceiling → CADP data-driven cognitive distillation → Prove better than roleplay (Exp 1) → Simulate real group dynamics (Exp 2)
- **Theory Positioning**: Bourdieu's Habitus serves as a **design blueprint** (not a tested theoretical claim). The three-dimensional mapping provides the granularity for CADP's distillation axes; we do not measure "habitus scores" or validate the sociological construct itself. The blueprint yields **one falsifiable prediction** tested in §5.8: the three CADP ablations dissociate across metric layers (i.e., the decomposition is non-arbitrary rather than decorative).

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
2. 实证贡献 (Exp 1): 13 conditions × 3 datasets × 4 models 系统对比（含 COLLEAGUE.SKILL 链式消融 + Clustering-Only 贡献隔离 + Constraint-Only mirror 消融 + Pop-Aligned 叠加测试）
3. 发现贡献 (Exp 2): 三真实社区的涌现动态

> Figure 1: 概览图

---

## Chapter 2: Related Work (~1.5 pages)
### 2.1 LLM-based Social Simulation
- Horton (2023), Park et al. (2023), Argyle et al. (2023)
- Park et al. (2024, *arXiv preprint* 2411.10109) — 大规模模拟 1,052 真实个体（interview-based persona），验证 generative agent 可信度；与 CADP 互补（interview-based vs behavioral-trace-based distillation）。**注**：截至 2026-06 仍为 arXiv 预印本（非 Nature 正式发表）；UIST 2023 的 25-agent "Generative Agents" 论文（Park et al. 2023）为不同工作
- Gap: 无人系统验证 persona vs 数据驱动方法的行为保真度

### 2.2 The Homogenization Problem
- Wu et al. (2025) — 诊断平均人格问题，无解决方案
- Li & Cheng (2026) — audience segmentation，top-down 描述性方法
- "The Chameleon's Limit" (arXiv:2604.24698) — 独立实证证据：persona collapse（agent 行为收敛到 modal pattern）是结构性失效模式
- "What Limits LLM-based Human Simulation: LLMs or Our Design?" (arXiv:2501.08579) — 系统拆解 LLM 模拟偏差来源（模型能力 vs prompting 设计），结论指向设计缺陷为主因；本文进一步将"设计缺陷"定位到 persona prompting 的结构性局限并提出 CADP 作为改进方案
- "LLM-Based Social Simulations Require a Boundary" (arXiv:2506.19806) — 提出 LLM 社会模拟的有效性边界问题；本文在 §7.4 Threats to Validity 中讨论 CADP 的适用边界
- Gap: segmentation 不保证个体级行为保真度；persona collapse 的根因（RLHF 压缩 + 描述性 prompt 缺乏行为规则）未被解决；现有 gap 诊断缺乏可操作的结构性解决方案

### 2.3 Population-Aligned Persona Generation（新增 — 最直接竞争者）

**方法概述**
- Population-Aligned Persona Generation (arXiv:2509.10127, Microsoft Research, 2025 *preprint*) — 用真实调查数据生成匹配人口**属性分布**的 persona 群体。生成目标是人口级 marginal alignment（per-attribute univariate 分布匹配），延续 Argyle et al. (2023) 的 silicon-samples 范式但扩展到 persona set 生成
- 同期相关工作：Survey-Derived Persona Prompt Collection (arXiv:2511.21722)、Deep Iterative Persona Alignment (preprints.org 2026) — 均属"分布匹配"族方法

**关键区别（CADP 的差异化 spine）**
- **层级区分**：Pop-Aligned 回答"有没有对的**类型**的人"（distributional validity, *types* of people present）；CADP 回答"这些人**做**得对不对"（behavioral fidelity, *behaviors* of those people）。前者是 *compositional* 维度，后者是 *interactional* 维度——两者正交，可叠加（§5.2 Condition 12 直接测试）
- **传递内容区分**：Pop-Aligned 传递属性标签（人口学 + 态度）；CADP 传递三维行为规则（Expression DNA / Mind Models / Anti-patterns）。属性标签是 *identity-level*，行为规则是 *rule-level*——前者对 RLHF 压缩无抵御能力，后者通过三层硬约束结构性绕过
- **挂载机制区分**：Pop-Aligned 通过 system prompt 静态注入；CADP 通过 dual-track .skill 文件 + three-tier enforcement 动态执行
- **验证目标区分**：Pop-Aligned 验证 marginal 分布匹配（univariate 或低维）；CADP 验证 emergent group dynamics（polarization, cascade, conflict escalation）+ held-out behavioral prediction

**证据基础与 Gap**
- Pop-Aligned 的有效性证据限于分布级指标；缺乏 (a) 行为级指标验证，(b) 群体交互动态验证，(c) 跨条件保真度对比
- **结构性局限**：正确的人口学构成不蕴含正确的交互模式。例：Reddit r/changemyview 中"大学生"和"中年技术从业者"的人口比例正确，不保证两者间的说服成功率分布正确——后者依赖 *behavioral rules*（如何论证、何时 award delta、何时拒绝让步），而这正是 CADP 蒸馏的对象
- **Gap 总结**：分布匹配（Pop-Aligned）≠ 行为保真度（CADP）。这一区分对应 §3.1 中 habitus 的语言性情（Pop-Aligned 不可触及）vs 认知图式 + 禁忌（CADP 蒸馏对象）的分层

**CADP 与 Pop-Aligned 的实验对比设计**
- §5.2 Condition 4（Pop-Aligned 复现）vs Condition 7（CADP Full）：预期 CADP 在 Macro Topology（分布级）接近 Pop-Aligned，但在 Micro Behavior、Predictive Fidelity 两层显著领先
- §5.2 Condition 12（Pop-Aligned + CADP 叠加）：测试可叠加性——若叠加显著优于纯 CADP → 两维度正交互补；若无明显增益 → CADP 的行为规则已隐含属性信息（"做对的事"已蕴含"是对的人"）

### 2.3.5 Sociological Framing of AI Behavior（新增）
- "The taste, class and habitus of generative AI chatbots" (Sage Journals, 2025) — 将 Bourdieu 的 habitus 概念应用于分析 LLM 的 taste/class 表现
- **与本文的关系**：该论文在分析层面使用 habitus 概念（解读 AI 输出中的阶层偏好）；本文在方法设计层面将 habitus 三维结构作为 **设计蓝图（design blueprint）**，并检验其启发的三维分解是否经得起消融解离检验（§5.8），但**不声称验证 Bourdieu 理论本身**。两篇论文的 habitus 使用层次不同：analytic lens vs design blueprint

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
### 3.1 Bourdieu's Habitus as a Design Blueprint
- Bourdieu's Habitus 三维结构（语言性情 / 认知图式 / 禁忌）作为 CADP 三维设计（Expression DNA / Mind Models / Anti-patterns）的**设计蓝图（design blueprint）**——不是装饰性引用，而是为三维分解提供 granularity 的来源。
- **主张界定**：本文**不**声称测量或验证 habitus 这一社会学构念本身。我们使用的是 habitus 三维分解的 *粒度*（三个正交的行为轴）作为 distillation 的合适 granularity；评估（§5）直接测量行为保真度，不涉及 habitus 的社会学构念。
- **可证伪的设计预测（§5.8 检验）**：若 habitus 启发的三维分解是 *有意义的 granularity* 而非任意切分，则三个消融（minus Expression DNA / minus Mind Models / minus Anti-patterns）应在**不同的 metric layer** 上产生最大损失，而非全部 load 在单一因子上。具体预测：
  - minus Expression DNA → Linguistics 层损失最大
  - minus Mind Models → Meso Dynamics 层损失最大
  - minus Anti-patterns → 冲突/极化相关指标（Macro E-I Polarization + Meso conflict escalation）损失最大
  - 若三组损失高度相关（加载到单一因子），则该分解的解释力被削弱——这是对"三维非任意"的直接检验，本文如实报告结果
- 与 "The taste, class and habitus of generative AI chatbots" (Sage, 2025) 的区别：该论文用 habitus 作为 **分析透镜（analytic lens）** 解读 AI 行为；本文用 habitus 三维结构作为 **方法设计蓝图（design blueprint）** 并检验其启发的分解是否经得起消融检验。两篇论文的 habitus 使用层次不同。

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

**方法学循环性防御声明**：聚类、编译、验证均基于同一社区数据，存在循环性质疑风险。本文通过四重机制防御：(1) 置换检验（§5.2 Condition 6 Shuffled）— 检测随机性；(2) 跨数据集迁移（§5.5 Wikipedia→Reddit）— 检测过拟合到特定社区；(3) Held-out 事件预测准确率（§5.3 Predictive Fidelity layer）— 检测方法是否捕获因果模式而非 spurious correlation；(4) 跨结构部分迁移（§5.5，共享分类器校准）— 在 community A 编译 .skill，部分迁移到结构不同的 community B（如 Reddit → GitHub：迁移可泛化组件 Expression DNA + 通用 Anti-patterns，含其校准的 θ_sem 语义阈值；平台特定的 Mind Models 与 Category C 行为分类器在 B 上重编译），若 fidelity 部分保持则说明捕获的是可泛化行为模式。Held-out 事件定义标准：由 2 名标注者独立编码争议性事件（conflict escalation / persuasion success / consensus formation），Cohen's κ ≥ 0.7 后取共识标签作为 ground truth。

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

> **Tier 1 高维校正说明**：对高维 embedding（d=384, `all-MiniLM-L6-v2`），"2σ boundary" 按 family-wise error rate 校正（Bonferroni），实际 per-dim-max z-score 阈值 ≈ 3.54σ（`alpha_per_dim = 2(1-Φ(2))/d`）。Naive 2σ 在 d=384 下期望 max|z| ≈ √(2 ln d) ≈ 3.45，会系统性过拒绝并虚增 CADP-Full 的 safe-template fallback 率。校正后保持 1-D 2σ 的 family-wise 显著性水平。
| Tier 2 | Mind Models | Pre-generation retrieval-augmented context injection (dynamic rule selection) | 生成前 |
| Tier 3 | Anti-patterns | **两阶段执行**：(a) Pre-gen 阶段—advisory reformulation injection（向 system prompt 注入 reformulation instruction + 记录 violation）；(b) Post-gen 阶段—**Forced Reformulation Protocol (§4.4.2)**：violation → block 当前输出 → diagnosis injection → constrained regeneration (max N_retry=3) → 若仍触发则降级为 safe-template fallback。Post-gen 阶段是真正的 hard-block-with-regeneration，作为结构性 RLHF override 的执行点 | 两阶段：pre-gen advisory + post-gen hard block |

> **Tier 2 与 Descriptive Persona system prompt 的技术区别**：Descriptive Persona 在 system prompt 中静态注入身份标签（一次性、不随对话上下文变化）。Tier 2 采用 **retrieval-augmented rule conditioning**：每轮根据当前对话状态（stance direction, conflict intensity, topic domain）从 Mind Models 规则库中动态检索最相关的 3-5 条推理模板注入 context。关键区别：(1) 动态 vs 静态——规则随上下文变化；(2) 条件化 vs 笼统——根据 agent 当前推理阶段选择匹配模板（如进入冲突阶段时检索"对抗性论证模板"而非"共识寻求模板"）；(3) Mind Models 包含从行为数据蒸馏的推理路径（如何从 A 推到 B），而非仅身份描述。若 §5.8 消融显示 minus Mind Models 下降不显著，则讨论 Tier 2 的角色可能主要为 Tier 1/3 提供 conditioning context（辅助功能），而非独立行为贡献。

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
- **α Sensitivity Analysis（新增到 §5）**：三个 pairwise 5×5 sweep（每次固定第三 tier 于 α=1.0），共 **75 cells** per (dataset, model)（详见 §5.6.5）；报告 sim-to-real gap 指标随 α 的变化曲线，确认最优 α 配置并验证 robustness
- **与消融实验的关系**：消融条件（minus Expression DNA / minus Mind Models / minus Anti-patterns）= 对应维度 α=0 且移除规则内容；α 调参是保留规则内容但调节执行强度。两者正交。

#### 4.4.4 模型适配策略（修订）
- **所有模型统一基线执行方案**：output filtering + re-prompting（Tier 1 embedding filter + Tier 3 forced reformulation via diagnosis injection）
- **开源模型增强（探索性）**：logit bias intervention 作为可选增强
  - 机制：对 anti-pattern token sequence 施加负 logit bias（bias 值通过 calibration set 回归确定）
  - **与现有工作的关系**：现有 constrained decoding 文献（JSON schema enforcement, grammar-guided decoding, safety refusal-and-retry 机制）处理的是**输出格式约束**或**安全过滤**。CADP 的 logit intervention 将约束对象从格式/安全扩展到**行为规则级别**（如论辩风格、交互模式约束），这一应用层面是新的，但底层机制（logit bias / constrained decoding）与现有 alignment 技术有结构性联系
  - 开源模型 logit intervention 定位为 **deferred to future work**（v1 release 未实现）。**注**：由于未实现，本文不声称 "negative finding"（未测试 ≠ 测试后为负）；§7.4 Threats to Validity 已将该方案列为效果不确定的 exploratory direction
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
- 30-50 轮/条件（**轮次设定依据**：从真实数据统计争议性事件的交互轮次中位数作为基准，sandbox 轮次设为基准的 1.5× 以允许动态涌现空间），重复 5-10 次

### 4.7 Baselines (扩展为 13 条件)
1. Vanilla LLM
2. Descriptive Persona
3. Segmentation Persona (复现 Li & Cheng 2026)
4. **Population-Aligned Persona (复现 arXiv:2509.10127)** — 最直接竞争者，属性分布匹配
5. **COLLEAGUE.SKILL (single-track, no enforcement)** — 直接前驱方法，隔离 dual-track + enforcement 的增量贡献
6. **Clustering-Only Descriptive Persona** — 共享 CADP 的 Step 1 聚类，但每聚类使用 descriptive persona 而非 .skill；隔离聚类贡献 vs 行为规则蒸馏贡献
7. CADP (Full)
8. CADP (Shuffled) — 置换检验（shuffle agent→skill assignment）
9. **CADP minus Expression DNA** — 消融
10. **CADP minus Mind Models** — 消融
11. **CADP minus Anti-patterns** — 消融（移除 Constraint Track）
12. **CADP Constraint-Only** — mirror 消融：仅保留 Constraint Track（Tier 3 ON，Capability Track 关闭）。与 11 互补：11 测试移除 Constraint Track 的损失，12 测试 Constraint Track 单独是否足够。两者共同构成 dual-track 必要性的双向证据
13. **Pop-Aligned + CADP** — 叠加条件：Pop-Aligned 选人 + CADP 赋予行为规则；测试可叠加性

> **13 条件设计说明**：原始设计为 12 条件。Code review (2026-06-16) 发现实际 config (`configs/exp1_full.yaml`) 实现了 13 条件，其中 `cadp_constraint_only` (条件 12) 是有价值的 mirror 消融——回答 "Constraint Track 单独是否足够" 这一与条件 11 ("移除 Constraint Track 损失") 互补的问题。本节正式将 13 条件纳入 outline。Cell 总数：13 × 3 datasets × 4 models × 5 repeats = 780 cells。

> **实验设计原则**：所有条件共享 Step 1 聚类结果（相同的 agent 分组结构），差异仅在于每组 agent 接收的 persona/skill 内容和 enforcement 机制。这隔离了"聚类结构"与"行为规则蒸馏"的贡献，避免混淆。
> **消融逻辑链**：COLLEAGUE.SKILL (single-track, no enforcement) → CADP minus Anti-patterns enforcement (dual-track, no enforcement) → CADP Full (dual-track + three-tier enforcement)，逐层隔离 structural contribution
> Figure 2: CADP Pipeline 流程图 (含 dual-track skill + three-tier enforcement)

---

## Chapter 5: Experiment 1 — Method Validation (~3 pages)
### 5.1 Setup
- 数据集: Wikipedia Talk Pages / Reddit r/changemyview / GitHub Issues
- 模型: GPT-4o / Claude 3.5 Sonnet / Llama-3-70B / Qwen-2.5-72B
- 总条件数: 13 × 3 × 4 = 156
- 重复: 每条件 5-10 次
- **样本量与可行性说明**：156 cells × 10 repeats = 1,560 simulation runs per metric；API 成本估算（GPT-4o + Claude 3.5）约 $3,000–7,000；开源模型推理（Llama-3-70B, Qwen-2.5-72B）需 4× A100 80GB，估算 300–500 GPU-hours。条件 10-13（Clustering-Only, COLLEAGUE.SKILL, Constraint-Only, Pop-Aligned+CADP）可在单数据集（Wikipedia）+ 双模型（GPT-4o + Llama-3-70B）上运行作为 reduced factorial，降低成本
- **Power analysis**：基于 pilot data（Wikipedia 单数据集）的 effect size 估计，Cohen's d ≥ 0.5 时 5 次重复即可达到 80% power (α=0.05)；若 pilot 显示 d < 0.5，则增加至 10 次重复
- **聚类共享原则**：所有 13 个条件使用相同的 Step 1 聚类结果（相同 agent 分组），差异仅在 persona/skill 内容和 enforcement 机制。这隔离聚类结构贡献与行为规则蒸馏贡献
- **Statistical analysis（确认性 vs 探索性声明）**：Confirmatory comparisons = CADP (Full) vs 每个 baseline 在 5 个 metric layer 上的逐层检验，层内 Bonferroni 校正；报告 effect size (Cohen's d) + 95% CI，不仅 p 值。消融（条件 9-12）、α-sweep（§5.6.5）、迁移测试（§5.5）、trigger calibration（§5.3.5）归为 exploratory，描述性报告、不做 family-wise 校正，结论措辞相应弱化（"suggest"/"indicate" 而非 "prove"）
- **可复现性 (Reproducibility)**：发布 (a) 完整 pipeline 代码 + nuwa-skill 编译器，(b) 化名（anonymized）聚合级 .skill 文件（不发布个体级，见 §7.5 dual-use），(c) 全部随机种子，(d) API 模型快照日期（GPT-4o / Claude 3.5 Sonnet 具体版本）与开源模型 commit hash（Llama-3-70B / Qwen-2.5-72B），(e) 标注协议 + 标注数据。数据集来自公开平台 API（CC-BY-SA / 平台 ToS 研究用途）

### 5.2 Baselines (扩展为 13 条件)
1. Vanilla LLM (无 persona)
2. Descriptive Persona (标准 system prompt)
3. Segmentation Persona (复现 Li & Cheng 2026)
4. **Population-Aligned Persona (复现 arXiv:2509.10127)** — 最直接竞争者，测试"属性分布匹配"能否达到"行为规则蒸馏"的保真度
5. **COLLEAGUE.SKILL (single-track, no enforcement)** — 直接前驱方法。在同一 sandbox 中用 nuwa-skill 框架编译 single-track .skill（仅 Capability Track），无三层 enforcement。隔离 dual-track + enforcement 的增量贡献
6. **Clustering-Only Descriptive Persona** — 共享 Step 1 聚类结果，每聚类使用描述性 persona（如"你是 Wikipedia 编辑者，属于 cluster A"）而非 .skill。隔离聚类结构贡献 vs 行为规则蒸馏贡献。若此条件已接近 CADP Full，则说明优势主要来自聚类而非 .skill
7. CADP (Full) — 三维完整 + three-tier enforcement
8. **CADP (Shuffled)** — 置换检验。**Shuffle 定义**：保持 agent 分组结构不变，但随机重分配 .skill 到错误聚类（将 cluster A 的 .skill 分配给 cluster B 的 agent）。测试"正确匹配"的重要性——若 shuffled 性能 ≈ random baseline，则说明正确 .skill 分配是关键；若 shuffled 仍优于 baseline，则说明 .skill 本身有 generic 价值
9. **CADP minus Expression DNA** — 去语言维度消融
10. **CADP minus Mind Models** — 去认知维度消融（同时检验 Tier 2 retrieval-augmented conditioning 的独立贡献）
11. **CADP minus Anti-patterns** — 去反模式消融 (测 RLHF override 假设)
12. **CADP Constraint-Only** — mirror 消融：仅保留 Constraint Track（Tier 3 ON，Capability Track 关闭）。与条件 11 互补：11 测试移除 Constraint Track 的损失，12 测试 Constraint Track 单独是否足够。两者共同构成 dual-track 必要性的双向证据
13. **Pop-Aligned + CADP** — 叠加条件：Pop-Aligned 人口属性选人 + CADP 行为规则蒸馏。测试可叠加性——若叠加显著优于纯 CADP，说明属性维度与行为维度互补；若无显著差异，说明行为规则已隐含属性信息

> 消融实验目的：隔离三维各自贡献，回应"维度选择是否arbitrary"的审稿质疑
> **消融逻辑链**（逐层隔离结构贡献）：
> - COLLEAGUE.SKILL (single-track) → CADP minus Anti-patterns enforcement → CADP Full：隔离 dual-track + enforcement 的增量
> - Clustering-Only → Descriptive Persona：隔离聚类贡献
> - CADP (Full) vs CADP (Shuffled)：验证正确 .skill 分配的必要性
> - Pop-Aligned + CADP vs CADP alone：测试属性/行为维度互补性
> 预期：Anti-patterns 移除对冲突/极化指标影响最大；Expression DNA 移除对语言指标影响最大；COLLEAGUE.SKILL 弱于 CADP minus Anti-patterns（说明 Constraint Track 结构本身有价值）
> 关键对比预期：CADP (Full) vs Pop-Aligned — 属性分布匹配能接近但无法达到行为规则级的保真度，差距在 Micro Behavior 和 Predictive Fidelity 层最显著

### 5.3 Five-Layer Evaluation Metrics
- **Macro Topology**: ΔQ Modularity, E-I Polarization Index, NED, **Coverage** (行为空间覆盖率, from Xiao et al. 2026)
  - Ground truth: 从真实社区同期交互日志中提取的网络结构（同一时间窗口的交互图）
- **Meso Dynamics**: Cascade Length Fit (KS-test), DTW, **Structural Fidelity** (交互网络结构相关, from Li & Cheng 2026)
  - Ground truth: 真实社区中已发生争议事件的 cascade 长度分布、时间序列
- **Micro Behavior**: Action Matrix Similarity (Frobenius), RSA, **Uniformity** (行为分布熵, 检测同质化), **Complexity** (跨agent行为方差)
  - Ground truth: 独立于 CADP 聚类的真实用户行为分布。**优先使用外部标注的行为类型分类**（Wu et al. 2025 audience segmentation 标签或人工标注的用户角色：moderator / provocateur / peacemaker / lurker）——`MetricsAggregator(role_labels_dir=...)` 检测 `data/role_labels/{dataset}.jsonl` 文件，存在时使用外部角色标签作为 ground truth，避免用 CADP 自己的聚类结果（循环依赖）。
  - **Fallback 路径（已实现）**：当外部标签文件不存在时，aggregator 退化为 Louvain 社区发现（从真实交互图推断）作为 ground-truth proxy，并记录到 `datasets_using_role_label_proxy`，供 §7.4 报告 per-dataset validity 差异。Louvain proxy 仍然独立于 CADP 的聚类（不存在循环依赖），但弱于外部角色标签——这是一个 acknowledged limitation 而非 spec 违反。Micro 层的 Frobenius / RSA / Uniformity / Complexity 度量直接比较 agent-by-action matrix，不强依赖 cluster 标签，因此 proxy fallback 的影响主要在 Macro E-I polarization 层
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
- **大数据集子采样**：为计算可行性（bootstrap 每轮重嵌入全部消息），对超过 2,000 threads 的数据集在 ARI bootstrap 前随机子采样到 2,000 threads；ARI variance 估计在子采样上统计充分

### 5.5 Cross-Dataset Transfer Test (新增)
- **同类型跨数据集迁移（方法论迁移 + 全组件迁移）**：
  - **全组件迁移**：在 Wikipedia 编译完整 .skill → 直接应用于 Reddit。预期：Expression DNA（语言模式）迁移良好；Mind Models 和 Anti-patterns 中平台特定规则（如 revert 行为相关规则）迁移后失效——报告各维度迁移后的 fidelity 保持率
  - **方法论迁移**：在 Wikipedia 上确定 CADP 的超参数（K 值、聚类权重、trigger 阈值 θ_sem），将这套参数迁移到 Reddit 上重新运行完整 CADP pipeline（包括重新编译 .skill）。测试的是"方法可泛化性"而非"规则可移植性"
- **跨结构部分迁移（共享分类器校准）**：在 Reddit r/changemyview 上编译 .skill → 部分迁移到 GitHub Issues。**迁移内容**：Expression DNA + 通用 Anti-patterns（每个 AP 携带其校准的 θ_sem 语义阈值，见 `schema.py` AntiPattern.trigger_semantic_threshold=0.85）。**在 GitHub 上重编译**：Mind Models、平台特定 Anti-patterns、以及 Category C 行为分类器（因其特征向量含平台特定信号如 delta_award_rate，不可跨平台直接迁移）。注：因目标侧重编译部分组件，此为部分迁移而非 zero-shot
  - **迁移范围界定**：仅迁移 Expression DNA（语言风格）+ 通用 Anti-patterns（如"人身攻击禁止"），不迁移平台特定规则（如 delta 机制、revert 规则）
  - GitHub 上重新编译平台特定的 Mind Models 和 Anti-patterns，但使用 Reddit 校准的 trigger classifier 参数
  - 若迁移后仍保持部分 fidelity → 证明 CADP 捕获的是可泛化行为模式（超越平台特定 artifact）
  - 这是对方法学循环性的最强防御：不同交互结构的社区间迁移排除了"过拟合到平台特征"的解释
- 报告：(1) 全组件迁移 vs 方法论迁移的 fidelity 差距，(2) 各维度迁移后保持率，(3) trigger classifier 跨域 Precision/Recall 下降程度
- 直接回应方法学循环性质疑

### 5.6 Human Evaluation (新增)
- 3 名领域专家盲评 100 条仿真对话
- 条件盲分配 (CADP vs Descriptive Persona vs Real)
- Cohen's κ ≥ 0.6 为可接受评分一致性
- 目的：外部效度补充，防止 metric overfitting

### 5.6.5 α Sensitivity Analysis (新增)
- **扫描设计**：三个 pairwise 5×5 sweep，每次固定第三个 tier 于 α=1.0：
  - α_expr × α_anti（fix α_mind=1.0）—— **primary**，回应 §3.2 anti-patterns 作为 RLHF override 的核心假设
  - α_expr × α_mind（fix α_anti=1.0）
  - α_mind × α_anti（fix α_expr=1.0）
  - 共 3 × 25 = **75 cells** per (dataset, model)。**不**做完整 5³=125（3 datasets × 4 models 下不可行）；pairwise-at-1.0 覆盖三维 cube 在 (1,1,1) 角的三个 2D 面。**范围限制**：三维同时取中间值（如 (0.5,0.5,0.5)）的内点未探索，作为 acknowledged scope limit
- **报告**：
  - α_expr vs α_anti 的独立影响曲线（Mind Models 固定为 α=1.0）
  - 三个 pairwise sweep 各自的最优 cell + 等高线（plateau vs 尖锐 peak）
  - 不同社区类型（Wikipedia / Reddit / GitHub）的最优 α 配置差异
  - robustness 检验：最优区域是否 plateau（`check_robustness` tolerance=0.05）而非尖锐 peak
- **目的**：证明 per-dimension α 的必要性（不同社区需要不同配置）并验证 enforcement 强度的可控性

### 5.6.7 Persona Collapse Stress Test (新增 — 直接回应 "The Chameleon's Limit")
- **目的**：纵向检验 CADP 的硬约束是否能在长交互链中抵御 persona collapse——回答 Chameleon's Limit (arXiv:2604.24698) 提出的结构性失效问题
- **协议**：50+ 轮交互（高于主实验 §5.1 的 30-50 轮），vanilla Descriptive Persona vs CADP Full，per-turn 测量 silhouette / Davies-Bouldin / behavioral entropy / persona embedding drift
- **预期**：Descriptive Persona 在 20-30 轮后出现 collapse 信号（silhouette 单调下降，entropy 收缩）；CADP 保持 plateau
- 详细设计：`docs/r4_persona_collapse_stress_test.md`

### 5.7 Results
- CADP (Full) 全面显著优于所有 baseline (13 条件对比)
- **Safe-template 分层报告**：所有 Linguistic 层指标按 `metadata.constraint_forced` 分层统计——safe-template 输出（Forced Reformulation Protocol fallback，§4.4.2 step 4）单独报告，不混入主指标均值。这对 `cadp_constraint_only`（条件 12）尤其关键：该条件 Tier 3 触发率高，safe-template 频率显著高于其他条件，若不分层会人为拉低 Linguistic 数字
- **COLLEAGUE.SKILL vs CADP 链式对比**：COLLEAGUE.SKILL (single-track) < CADP minus Anti-patterns enforcement < CADP Full，逐层隔离结构贡献
- **Clustering-Only vs Descriptive Persona vs CADP**：隔离聚类贡献——若 Clustering-Only 显著优于 Descriptive Persona，需报告聚类占 CADP 优势的比例
- 消融分析: 各维度独立贡献量化
- 关键对比: CADP vs Segmentation Persona 逐维度差异；CADP vs Population-Aligned Persona 的行为保真度差异
- **Pop-Aligned + CADP 叠加效果**：是否显著优于纯 CADP（属性/行为互补性）
- 置换检验: Shuffled（agent→skill 重新分配）显著弱于 Full
- 跨数据集迁移: 全组件迁移 vs 方法论迁移的 Wikipedia→Reddit 结果；Reddit→GitHub 跨结构部分迁移结果（分维度保持率）
- 预测性保真度: held-out 事件预测准确率
- Trigger calibration: per-category P/R/F1 + 跨域迁移性能
- α Sensitivity: per-dimension α 曲线 + 不同社区最优配置
- 跨模型/跨数据集一致性
- **Human evaluation 结果**：3 名专家盲评（Cohen's κ ≥ 0.6）中 CADP 的辨识度——专家能否区分 CADP 仿真 vs 真实交互（作为外部效度的核心证据，在主结果中报告）

> Table 1: 主结果表 (13 conditions × 5 metric layers)
> Table 2: 消融结果表 (3 维度 × 5 metric layers) + COLLEAGUE.SKILL 链式对比
> Table 2b: Clustering-Only vs Descriptive vs CADP（聚类贡献隔离）
> Figure 3: 雷达图 — 13 条件 5 层指标对比
> Figure 4: 交互网络可视化对比

### 5.8 Analysis
- 各维度贡献分析 (基于消融)
- **三维 ↔ metric-layer 解离检验（验证 §3.1 预测）**：检验 minus Expression DNA / minus Mind Models / minus Anti-patterns 三者的 per-layer 损失是否解离——预测各自在不同 metric layer 达到峰值损失（EDNA→Linguistics / MM→Meso / AP→冲突极化）。报告三组 per-layer 损失的相关矩阵：若 off-diagonal 相关高（三维损失加载到单一因子）则如实承认分解解释力削弱；这是对 habitus 启发分解"非任意性"的直接检验
- **聚类贡献 vs 行为规则贡献的分解**：Clustering-Only 条件的性能定位了聚类的独立贡献上限；CADP Full - Clustering-Only = 行为规则蒸馏的净贡献
- **Tier 2 独立贡献检验**：CADP minus Mind Models 的下降幅度——若不显著，讨论 Tier 2 的 retrieval-augmented conditioning 是否主要为 Tier 1/3 提供 context（辅助角色）
- Descriptive Persona 在哪些指标上最接近 CADP
- **CADP vs Pop-Aligned 深入对比**：在哪些 metric layer 差距最大/最小？Pop-Aligned 在 Macro Topology（分布级）可能接近，但在 Micro Behavior 和 Predictive Fidelity（行为级）预期显著落后——量化"属性匹配 ≠ 行为匹配"
- **COLLEAGUE.SKILL → CADP 的增量来源**：single-track vs dual-track 的差距来自 Capability+Constraint 结构（dual-track 本身）还是 enforcement 机制（three-tier）？通过 COLLEAGUE.SKILL vs CADP-minus-Anti-patterns-enforcement vs CADP Full 三点对比拆解
- **Pop-Aligned + CADP 叠加效果解读**：若叠加无显著增益，说明 CADP 的行为规则已隐含人口属性信息；若有增益，说明两维度正交互补
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
- CADP agents 交互 30-50 轮（**轮次对齐**：根据 pilot data 确定大多数争议性事件在真实社区中的交互轮次中位数，sandbox 轮次与之对齐；若真实事件中位数交互量为 M 轮，sandbox 设为 ⌈M × 1.5⌉ 以允许动态涌现空间）
- **对比条件（4 条件）**：真实历史 vs CADP (Full) vs Pop-Aligned Persona vs **CADP minus Anti-patterns**。Pop-Aligned 保留为最强竞品（Descriptive Persona 移至 §5 完整对比，Exp 2 聚焦最强 baseline）；**CADP minus Anti-patterns 是机制验证条件**——若移除 Constraint Track 后涌现动态（极化、冲突升级）显著退化甚至消失，则在动态轨迹层面验证 §3.1 的预测（Anti-patterns → 冲突/极化），把 Exp 2 的发现从相关性（"CADP 工作"）升级为机制性（"Anti-patterns 驱动涌现"）
- **Scale test（robustness，非头条）**：主实验 N≈30（K=3-5 聚类，每聚类 6-10 agents）。附加 N≈100 条件（K=10-15 聚类，每聚类 7-10 agents——聚类数等比增加以避免同 .skill 多 agent 导致的行为同质化）作为 **robustness check / appendix**，回答"fidelity 是否为 30-agent 伪迹"，不作为一等发现

### 6.3 Platform-Specific Action Space Enforcement
每个平台强制执行真实交互拓扑和动作空间。`PlatformTopology.select_reply_target`
（两阶段：planner 输出 `target_msg_id` 提示 → topology 按平台规则
裁决/修正）保证生成的每条消息挂载到符合平台拓扑的父节点上，
而不是统一压扁成"上一条消息的回复"：
- **Wikipedia**: 编辑树结构 (edit → revert → re-edit)；动作空间: edit / revert / discuss / report
  - `EDIT` 修改文章本体 → `parent=None`；`REVERT` 目标 = 最近一次他人 EDIT；
    `DISCUSS`/`REPORT` 目标 = 最近一条他人消息
- **Reddit r/changemyview**: 线程回复 + delta 机制；动作空间: reply / award delta / counter-argue / block
  - `AWARD_DELTA` 目标 = 反驳过自己的那条 `COUNTER_ARGUE`；
    `COUNTER_ARGUE` 目标 = 最近的非 BLOCK 他人消息；
    `REPLY`/`BLOCK` = 标准线程回复
- **GitHub Issues**: Issue 生命周期 (open → comment → close/reopen)；动作空间: comment / label / close / reopen / assign
  - `LABEL`/`CLOSE`/`REOPEN`/`ASSIGN` 是 issue 级事件 → `parent=None`；
    `COMMENT` 目标 = 最近的他人 COMMENT

### 6.4 Temporal Trajectory Analysis (新增)
- 不仅对比终态，还对比**演化轨迹**
- 极化指数按轮次画演化曲线，DTW 对齐真实时间线
- 追踪关键事件时机：首次冲突、首次说服成功的出现轮次 vs 真实时间点
- 目的：终态匹配 ≠ 过程匹配，轨迹对比更严格

### 6.4.5 Emergence Mechanism Ablation (新增)
- **CADP Full vs CADP minus Anti-patterns 的涌现轨迹对比**：在极化指数演化曲线、冲突升级 timing、说服级联深度上对比两者
- **预测（落地 §3.1）**：移除 Anti-patterns 后，agent 在冲突场景趋向 RLHF 妥协 → 极化无法涌现、冲突升级被过早平息、说服级联变浅；CADP Full 保持与真实历史接近的轨迹
- **结果解读**：若 minus Anti-patterns 的涌现动态显著偏离真实历史且趋向"扁平化"（persona collapse 的群体级表现），则在动态轨迹层面验证 Anti-patterns 作为 RLHF override 的机制；若退化不明显，如实报告并讨论（可能的平台/场景依赖）

### 6.5 Results by Dataset
- **Wikipedia**: 编辑冲突动态，极化指数演化曲线，编辑战 escalation pattern
- **Reddit**: 说服成功率 (delta award rate)，对抗网络结构，反驳链深度
- **GitHub**: 技术共识形成路径，群体分化时间线

### 6.6 Cross-Dataset Comparison
- 极化动态对比 (轨迹层面)
- CADP Full vs CADP minus Anti-patterns：Constraint Track 让 agent 群体"看见"了什么涌现动态（极化、冲突升级）、移除后又"看不见"了什么
- Scale test (robustness): N=30 vs N=100 fidelity 变化

> Figure 5: 极化指数时间演化曲线 (真实 vs CADP Full vs Pop-Aligned vs CADP minus Anti-patterns, 三平台)
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
- vs Park et al. (2024, arXiv:2411.10109, *preprint*, 1,052 individuals): interview-based vs behavioral-trace-based distillation；interview 依赖主动参与，CADP 可大规模从被动数据蒸馏
- vs "Habitus of GenAI" (Sage, 2025): 分析层面 vs 方法设计层面使用 habitus 概念

### 7.4 Threats to Validity（扩展为 Threats to Validity）
- **方法学循环性**：聚类→编译→验证基于同一社区数据。四重防御：(1) 置换检验（shuffle agent→skill assignment），(2) 跨数据集迁移 (Wikipedia→Reddit, 全组件 + 方法论两种迁移模式)，(3) held-out 事件预测（带标注者间一致性编码），(4) 跨结构部分迁移（Reddit→GitHub，仅迁移可迁移组件 + 平台特定规则重编译；因目标侧重编译，非 zero-shot）。四者一致指向 CADP 优势时循环性影响可控。
- **与 "What Limits LLM Simulation" (arXiv:2501.08579) 的关系**：该论文将 sim-to-real gap 归因为"LLMs 或 Our Design"。本文从 design 角度回应（persona prompting ceiling），但未排除 LLM 能力限制的交互效应——若更强大模型自然缩小 gap，CADP 的边际贡献可能随模型规模递减。本文通过 4 模型对比部分控制此威胁，但未来模型的能力提升可能改变结论。
- **与 LLM Simulation Boundary (arXiv:2506.19806) 的关系**：该论文提出 LLM 模拟的有效性边界。CADP 的适用边界：当社区行为高度依赖平台外知识（如线下社会关系、跨平台历史）时，仅从单平台行为轨迹蒸馏的 .skill 可能不足。
- **与 Population-Aligned Persona 的区分边界**：Pop-Aligned (arXiv:2509.10127) 做属性分布匹配，CADP 做行为规则蒸馏。两者非替代关系——§5.2 Condition 12 测试叠加效果（Pop-Aligned + CADP），实验结果将决定两者是互补还是冗余。
- **Logit steering 可行性限制**：行为规则级 logit intervention 与现有 constrained decoding 有结构性联系，应用层面（行为规则 vs 格式/安全）是新的。开源模型增强方案效果不确定，定位为 exploratory contribution。
- **Anti-pattern trigger 校准依赖**：trigger 阈值需 per-dataset 校准（§4.4.1），跨域迁移时 Precision/Recall 会下降——§5.5 报告具体下降程度。
- **Anti-patterns 编码社区偏见的风险**：CADP 从真实社区行为中蒸馏 anti-patterns，可能忠实地再现社区中的偏见性规范（如隐性歧视行为模式）。这是 "fidelity vs. ethics" 的结构性张力。**当前 release 不实现自动化 bias audit**（留作 future work，见 §7.5），所有 fidelity 数字均为 unaudited——即包含社区偏见的忠实再现，本文如实报告而非掩盖。
- **Bourdieu 框架的可证伪性**：本文不验证 habitus 社会学构念本身，但 §3.1 提出一个 *可证伪的设计预测*——三维消融应在不同 metric layer 上解离（而非加载到单一因子）。该预测在 §5.8 检验：若三组损失高度相关则如实承认分解解释力削弱。这是对"三维分解非任意"的检验，**不**构成对 Bourdieu 理论本身的验证。
- **模型依赖**：4 模型验证覆盖主流 API + 开源，但未测试 smaller models (<7B)。
- **平台覆盖**：仅英文平台，跨语言行为模式差异未验证。
- **时间窗口**：训练数据为特定时间段的交互快照，行为模式的时间漂移未建模。

### 7.5 Ethical Considerations（扩展）
- **IRB 与数据合规**：所有数据来自公开平台 API（CC-BY-SA / 平台 ToS 允许研究用途）；已通过 IRB 审查（或豁免）
- **PII 处理**：用户名、IP 引用、个人链接等 PII 在预处理阶段移除
- **Anti-patterns 与社区偏见**：CADP 从真实社区行为中蒸馏 anti-patterns——若社区存在系统性偏见（性别刻板印象、种族偏见行为模式等），CADP 会忠实地编码这些偏见，这是 "behavioral fidelity" 的直接后果，构成 fidelity 与 ethics 的结构性张力。
  - **当前 release 状态**：本文 **不** 实现自动化 bias audit；所有保真度数字均为 **unaudited**（包含社区偏见的忠实再现）。本文如实报告此状态，而非声称已做伦理审计。
  - **Future work**：规划的 bias audit step（编译阶段检测 anti-patterns 是否含 protected class 相关规则 + 人工审核标记项）及 audited/unaudited 双数字报告机制（以 audited 为 headline、unaudited 为参考上界，fidelity 不凌驾于伦理审计之上）留作未来工作。
  - **限制**：即便未来实现 bias audit，也无法覆盖所有隐性偏见，报告为 acknowledged limitation。
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
| Table 1 | 13 conditions × 5 metric layers main results | 5.7 |
| Table 2 | Ablation + COLLEAGUE.SKILL chain comparison (3 dimensions + single-track baseline × 5 metric layers) | 5.7 |
| Table 2b | Clustering contribution isolation (Descriptive vs Clustering-Only vs CADP) | 5.7 |
| Table 3 | Positioning comparison (5 methods × 7 dims) | 2.5 |
| Table 4 | Three-tier enforcement mechanism + trigger formalization | 4.4 |
| Table 5 | Trigger calibration results (P/R/F1 per category × 3 datasets) | 5.3.5 |
| Table 6 | α Sensitivity: 3 pairwise 5×5 sweeps × metric layers (75 cells, §5.6.5) | 5.6.5 |
| Figure 1 | Overview diagram | 1.5 |
| Figure 2 | CADP Pipeline flowchart (dual-track skill + three-tier enforcement) | 4.1 |
| Figure 3 | Radar chart — 13 conditions × 5 metric layers | 5.7 |
| Figure 4 | Interaction network visualization comparison | 5.7 |
| Figure 5 | Polarization index time evolution curves (real vs CADP vs baseline, 3 platforms) | 6.5 |
| Figure 6 | Simulated interaction network snapshots (multi-timepoint) | 6.6 |
| Figure 7 | Key event timing comparison (CADP vs real vs baseline) | 6.6 |
| Figure 8 | α Sensitivity heatmaps (per-dimension α × 5 metric layers, per dataset) | 5.6.5 |
| Figure 9 | Cross-structure transfer fidelity (Reddit→GitHub) | 5.5 |
