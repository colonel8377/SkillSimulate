# Paper Outline: Community-Trace Behavioral Skills

## Metadata
- **Title**: Community-Trace Behavioral Skills for LLM Social Simulation: A Feasibility Study of Distillation and Inference-Time Filtering
- **Target Venue**: ICWSM / WWW (10-12 pages)
- **Format**: English, LaTeX, ACM double-column (WWW/`acmart` template)
- **Core Narrative**: Persona prompts may carry useful identity information yet provide limited control over interactive behavior. CADP organizes community traces as reusable behavioral skills. Exp1 asks a narrow feasibility question: with identical Nuwa skill content, does retrieval plus filter-retry improve fidelity beyond advisory-only prompting on DeepSeek Flash × Wikipedia?
- **Headline Claim**: This single-model, single-community study tests methodological feasibility. It does not establish cross-model generality, stable natural archetypes, recovery of emergent group dynamics, or a weights-level solution to persona collapse.
- **Theory Positioning**: Bourdieu's habitus is retained only as design inspiration in Discussion. It is not a contribution or tested construct in the feasibility stage; any three-dimensional dissociation claim requires later ablations.
- **Enforcement Honesty Caveat**: Chameleon's Limit provides suggestive evidence from thinking/non-thinking comparisons that extended reasoning alone may not recover collapsed persona variation. CADP operates only at prompt/filter time and is evaluated as an inference-time intervention.

---

## Chapter 1: Introduction (~1 page)
### 1.1 Background: LLM-based Social Simulation
- 社会科学家/政策制定者转向 LLM agent 模拟预测群体行为
- 代表性工作概述（Argyle et al. 2023; Park et al. 2023; Horton 2023）

### 1.2 The Sim-to-Real Gap
- LLM agents 产出交互系统性偏离真实人类动态
- 表现：行为同质化、冲突缺失、极化无法涌现、说服失败
- 引用 Wu et al. (2025) 的问题诊断 + "The Chameleon's Limit" (arXiv:2604.24698) 的 persona collapse 独立实证证据
- 引用 "LLM-based Human Simulations Have Not Yet Been Reliable" (arXiv:2501.08579) — 该论文同时讨论模型能力与 simulation design；本文只检验 design-side 的一个干预

### 1.3 Motivation: Limits of Persona-Only Conditioning
- Descriptive persona 只传递身份标签而非行为规则
- "Average Persona Problem" (Qin et al. 2026)
- Persona collapse 可能同时来自模型训练与 simulation design；本文只检验 design-side 的 inference-time intervention，不作单一根因断言
- 引用 Cognitive Heuristics 论证 (Zhu & Heydari 2026)

### 1.4 Our Approach: CADP — Distilled Behavioral Skills + Inference-Time Filtering
**核心论点（central thesis）**：skill 内容与 skill 执行机制必须分开识别。`cadp_advisory_nuwa` 与 `cadp_full_nuwa` 使用同一 Nuwa skill；二者差异仅为 dynamic retrieval + Tier-1/Tier-3 filter-retry。当前 8-round gate 中 `reflection_interval=10`，reflection 不触发，因此不属于核心处理差异。Exp1 用这一 content-matched comparison 判断方法是否值得进入更大规模研究。
- 三维蒸馏（Expression DNA / Mind Models / Anti-patterns）为行为规则提供 granularity。**CADP 借用 nuwa-skill 的 5-layer 结构模板**（Expression DNA / Mental models / Decision heuristics / Anti-patterns / Honest boundaries）应用到社区聚类 archetype——不是用 nuwa 蒸馏 public figure，而是用 nuwa 的结构模板组织 WikiConv 行为轨迹。colleague-skill 6-layer persona（含 Work Skill + Correction Log）作为 methodology comparison baseline（§5.2 条件 6），不作为 CADP base structure
- **Inference-time 三层 filter-retry 执行**（pre-gen retrieval 注入 / post-gen embedding filter / post-gen trigger block + regeneration）是被检验的 supporting mechanism，**不是 weights-level hard constraint**。Chameleon's Limit 的结果只作为相关机制证据，不作普遍因果证明。
- **可证伪预测**：Full 相对 advisory-only 应在三个非重复 metric families 中至少两个达到预设最小改善，并通过 throughput、safe-template 与 action-text consistency guards；否则停止 method-led framing。

### 1.5 Contributions
1. **方法原型**：把社区行为轨迹组织为可检查的 behavioral skill，并实现 retrieval + filter-retry 执行路径。
2. **受控可行性评估**：在同一 Nuwa skill 内容下比较 advisory-only 与 Full，避免把信息量和执行机制混为一谈。
3. **诊断贡献**：以 action、interaction、independent linguistics 三个 metric families 和执行质量 guards 给出明确 GO/STOP。跨模型、跨数据集、机制消融、群体涌现与理论检验均为 GO 后工作。

> Figure 1: 概览图

---

## Chapter 2: Related Work (~1.5 pages)
### 2.1 LLM-based Social Simulation
- Horton (2023), Park et al. (2023), Argyle et al. (2023)
- Park et al. (2024, *arXiv preprint* 2411.10109) — 大规模模拟 1,052 真实个体（interview-based persona），验证 generative agent 可信度；与 CADP 互补（interview-based vs behavioral-trace-based distillation）。**注**：截至 2026-06 仍为 arXiv 预印本（非 Nature 正式发表）；UIST 2023 的 25-agent "Generative Agents" 论文（Park et al. 2023）为不同工作
- **CCP（Schwager et al., arXiv:2602.22752; WASSA/EACL 2026）**：用 Conditioned Comment Prediction 将生成评论与真实数字轨迹比较；其结果支持 authentic behavioral histories 优先于 generated biographies，并揭示 form/content decoupling。它与本文共享 trace-based operational-validity 动机，但研究单用户评论预测与 SFT，不研究多 agent 交互或 inference-time filtering。
- **MiroBench（Yu et al., arXiv:2606.14715, preprint）**：以 4,292 个 Reddit threads 比较 repetition/uniformity、narrative content、toxicity/aggression 与 structural complexity；五模型结果显示 prompt-only improvement 有限。它是本文多族 evaluation 设计的最接近 benchmark，但不蒸馏 community skills，也不做 content-matched enforcement comparison。
- Gap 收窄为：现有工作已评估 trace-conditioned prediction 与 discussion realism；仍缺少在**静态 skill 内容完全相同**时，对 advisory-only 与 retrieval/filter-retry 的受控可行性检验。

### 2.2 The Homogenization Problem
- **"LLM-Based Social Simulations Require a Boundary" (Wu et al. 2025, arXiv:2506.19806)** — 诊断平均人格问题并划定 LLM 社会模拟的有效性边界，无解决方案。**勘误**：outline 早期版本曾将 "Wu et al. 2025" 与 arXiv:2506.19806 当作两篇独立工作，实为同一篇（经 arXiv:2509.10127 参考文献表确认）
- Qin et al. (2026, arXiv:2604.06663) — audience segmentation，top-down 描述性方法。**勘误**：第一作者为 Qin，非早期版本的 "Li & Cheng"
- **"The Chameleon's Limit" (arXiv:2604.24698, Xiao et al., 2026-04)** — 10 个测试 LLM 在不同域呈现 persona collapse，并基于 Qwen thinking/non-thinking 对比提出 collapse **可能主要驻留于权重而非 reasoning chain**。本文将其作为机制证据而非已完成的普遍因果证明
- **"LLM-based Human Simulations Have Not Yet Been Reliable" (arXiv:2501.08579)** — 系统拆解模型能力与 simulation-design 两类偏差来源；CADP 是 design-side 的一个经验性干预，不声称排除模型能力限制
- Gap: segmentation 与 persona enrichment 不自动保证 interactional fidelity；persona collapse 的根因尚未完成因果识别。本文只测试一个可操作的 inference-time intervention，不宣称解决根因。

### 2.2.5 Two Remediation Levers: Description vs Constraint（新增 — 核心定位框架）

本文用**两个正交杠杆**组织整个补救方法谱系，这是 CADP 定位的主轴：

- **杠杆 1 — Description（说什么）**：调整 persona 的内容丰富度。所有现有竞品都在此杠杆上：
  - Descriptive persona（身份/属性标签）
  - Segmentation（Qin et al. 2026，描述性分割标识符）
  - Population-Aligned Persona（arXiv:2509.10127，Importance Sampling + population alignment；本文没有完整复现该 pipeline）
  - **PersonaEvolve / PEvo（Wang et al., EMNLP 2025 Main; arXiv:2509.16457）** — 以 PEBA 分布匹配框架迭代改写 persona，并在 high-stakes crowd simulation 中优于 explicit-instruction baselines；它构成强竞争证据，但优化对象与本文的社区轨迹、多 agent discussion fidelity 不同。
  - **PEP — Persona Ecosystem Playground（arXiv:2603.03140）** — RAG 生成描述性对话 persona + 软 RQE 阈值修订
  - **"Scaling Law in LLM Simulated Personality"（arXiv:2510.11734）** ⚠️ **最强威胁**：随 persona 细节增加，到人类 Big Five 年龄曲线的欧氏距离 70.25→63.45→51.21→23.75 单调下降，主张 "more detailed persona profile is all you need，无需 task-specific intervention"。（注：仅基于单模型自报问卷，且被 Zierahn et al. 2026, arXiv:2603.19030 反驳）
  - **"LLM Generated Persona is a Promise with a Catch"（NeurIPS 2025 Position, arXiv:2503.16527）** — Meta Personas（无 LLM 生成内容）对齐最好；Descriptive/Generative Personas（LLM 内容越多）偏差越大，跨 6 个 LLM、3 个选举周期普适。**此结果动摇"内容越丰富越真实"的前提** → CADP 必须论证 enforcement ≠ 再加 LLM 内容，而是截断生成分布（见 §3.2）
- **杠杆 2 — Inference-Time Intervention（怎么约束生成）**：在生成前后注入/过滤行为规则。CADP 的实现是 **filter-retry enforcement**（retrieval-augmented rule injection + post-generation embedding/trigger filter + constrained regeneration）。截至 2026-07 的检索未发现与本文完全相同的 community-trace skill + content-matched advisory/filter comparison；这是有时效限制的定位，不写成“首次”或穷尽性结论。
- **重要诚实声明**：杠杆 2 在本 feasibility study 中以 **filter-retry** 实现，**非 weights-level hard constraint**。Chameleon's Limit 的 thinking/non-thinking 结果只提示权重级因素可能重要；本文不把它表述为已完成的因果证明，也不声称 filter-retry 结构性克服 RLHF attractor。

**Gap**：杠杆 2（inference-time behavioral intervention）在 social simulation 域仍缺乏系统验证。Viability gate 使用项目内部的 **Rich Cluster Narrative** 作最强静态描述对照；它受 Scaling-Law 启发但不是该论文的忠实复现。CADP 若不能稳定击败该对照，则停止 method-led framing

### 2.3 Population-Aligned Persona Generation（新增 — 最直接竞争者）

**方法概述**
- Population-Aligned Persona Generation (arXiv:2509.10127, Microsoft Research, 2025 *preprint*) — 用真实调查数据生成匹配人口**属性分布**的 persona 群体。生成目标是人口级 marginal alignment（per-attribute univariate 分布匹配），延续 Argyle et al. (2023) 的 silicon-samples 范式但扩展到 persona set 生成
- 同期相关工作：Survey-Derived Persona Prompt Collection (arXiv:2511.21722)、Deep Iterative Persona Alignment (preprints.org 2026) — 均属"分布匹配"族方法

**关键区别（CADP 的差异化 spine）**
- **层级区分**：Pop-Aligned 回答"有没有对的**类型**的人"（distributional validity, *types* of people present）；CADP 回答"这些人**做**得对不对"（behavioral fidelity, *behaviors* of those people）。前者是 *compositional* 维度，后者是 *interactional* 维度——两者正交，可叠加（§5.2 Condition 12 直接测试）
- **传递内容区分**：Pop-Aligned 传递属性标签（人口学 + 态度）；CADP 传递三维行为规则（Expression DNA / Mind Models / Anti-patterns）。属性标签是 *identity-level*，行为规则是 *rule-level*——前者对 RLHF 压缩无抵御能力，后者通过三层 filter-retry 在 inference 阶段截断违规输出
- **挂载机制区分**：Pop-Aligned 通过 system prompt 静态注入；CADP 通过 nuwa 5-layer .skill 文件 + three-tier filter-retry 动态执行
- **验证目标区分**：Pop-Aligned 验证 marginal 分布匹配（univariate 或低维）；CADP 验证 emergent group dynamics（polarization, cascade, conflict escalation）+ 微观行为保真度

**证据基础与 Gap**
- Pop-Aligned 的有效性证据限于分布级指标；缺乏 (a) 行为级指标验证，(b) 群体交互动态验证，(c) 跨条件保真度对比
- **结构性局限**：正确的人口学构成不蕴含正确的交互模式。例：Reddit r/changemyview 中"大学生"和"中年技术从业者"的人口比例正确，不保证两者间的说服成功率分布正确——后者依赖 *behavioral rules*（如何论证、何时 award delta、何时拒绝让步），而这正是 CADP 蒸馏的对象
- **Gap 总结**：分布匹配（Pop-Aligned）≠ 行为保真度（CADP）。这一区分对应 §3.1 中 habitus 的语言性情（Pop-Aligned 不可触及）vs 认知图式 + 禁忌（CADP 蒸馏对象）的分层

**CADP 与 population-alignment 思路的实验对比设计**
- §5.2 的 `cluster_stat_aligned` 内部基线（代码标识暂保留 `pop_aligned`）仅从 CADP cluster statistics 采样属性；它不是 arXiv:2509.10127 的忠实复现，结果只能支持对“cluster-stat description”而非对原论文方法的比较
- §5.2 Condition 12（Pop-Aligned + CADP 叠加）：测试可叠加性——若叠加显著优于纯 CADP → 两维度正交互补；若无明显增益 → CADP 的行为规则已隐含属性信息（"做对的事"已蕴含"是对的人"）

### 2.3.5 Sociological Framing of AI Behavior（新增）
- "The taste, class and habitus of generative AI chatbots" (Sage Journals, 2025) — 将 Bourdieu 的 habitus 概念应用于分析 LLM 的 taste/class 表现
- **与本文的关系**：该论文在分析层面使用 habitus 概念（解读 AI 输出中的阶层偏好）；本文在方法设计层面将 habitus 三维结构作为 **设计蓝图（design blueprint）**，并检验其启发的三维分解是否经得起消融解离检验（§5.8），但**不声称验证 Bourdieu 理论本身**。两篇论文的 habitus 使用层次不同：analytic lens vs design blueprint

### 2.4 Behavioral Distillation & Cognitive Cloning
- COLLEAGUE.SKILL (Zhou et al. 2026, arXiv:2605.31264) — 面向个人工具，不声称行为保真度。**勘误**：经文献核实，COLLEAGUE.SKILL **本身是 dual-track**（capability track: practices / mental models / decision heuristics + bounded behavior track: communication style / interaction rules / correction history），早期版本"single-track"表述有误。**v1 reframe（2026-07-08）**：CADP 不再以 COLLEAGUE.SKILL 作 base structure。CADP 借用 **nuwa-skill 5-layer 模板**（Expression DNA / Mental models / Decision heuristics / Anti-patterns / Honest boundaries），因 nuwa 结构与 CADP 三维一对一对映、且 anti-patterns 层带结构化触发器（每 pattern 配 3 条真实引用，§4.4.1 trigger 校准直接可用）。COLLEAGUE 6-layer persona（含 Work Skill + Correction Log）作 **methodology comparison baseline**（§5.2 条件 6）——回答 "CADP framework 是否依赖特定 distillation 结构"。本文 baseline `colleague_skill_full` 为完整 6-layer colleague 蒸馏（capability + advisory behavior），**不消融、不裁剪**，作为 methodology head-to-head 对照
- nuwa-skill 框架（github.com/alchaincyf/nuwa-skill）— 5-layer cognitive OS 蒸馏（Identity / Mental models / Decision heuristics / Expression DNA / Anti-patterns + Honest boundaries），目标 = public figure perspective skill。**CADP 借用 nuwa 5-layer 模板**作 base structure（§4.3），应用到社区聚类 archetype。两者关系：结构继承 + 应用域扩展（public figure → community archetype）+ filter-retry enforcement + habitus 三维分类
- Zhu & Heydari (2026) — 理论推导，无实证
- Gap: 最接近工作覆盖 trace-conditioned prediction、persona optimization 或 realism benchmarking；本文只主张其窄增量——community-trace skill 的 advisory/filter content-matched feasibility test。

### 2.5 Positioning Summary Table

| 维度 | Descriptive Persona | Segmentation | Pop-Aligned (2509.10127) | nuwa-skill | COLLEAGUE.SKILL | CADP (Ours) |
|------|--------------------|-------------|--------------------------|-----------|-----------------|-------------|
| 数据来源 | 人工/调查 | 人口学数据 | 调查数据 | 公开人物研究 | 个人行为轨迹 | 社区行为轨迹 |
| 传递内容 | 身份标签 | 人口学+心理标签 | 人口分布属性 | 5-layer 认知OS（含显式 anti-patterns）| 6-layer persona + Work Skill | 5-layer 行为规则（继承 nuwa）|
| 挂载方式 | system prompt | system prompt | system prompt | .skill 文件 | .skill 文件 | .skill filter-retry |
| 模拟目标 | 个体/群体 | 群体分布 | 群体分布匹配 | 个人/名人工具 | 个人工具 | 群体交互动态 |
| 约束机制 | 无 | 无 | 无 | advisory（Honest boundaries）| advisory + Correction Log | 三层 filter-retry (per-dim α) |
| RLHF override | ✗ | ✗ | ✗ | ✗ | 部分（negative examples + Correction）| 部分（filter-retry 不碰权重，§7.4 acknowledged） |
| 保真度验证 | 无 | 分布级 | 分布级 | 无 | 无 | 四层级系统验证 |

> **Table 说明**：✗/✓/部分 的区分旨在反映 continuum 而非 binary。CADP 的 base structure 继承自 nuwa-skill（5-layer 三维对映），colleague-skill 6-layer 作 methodology comparison baseline（条件 6）。CADP 相对两者的增量：(a) 社区级 behavioral-trace 应用域；(b) 三层 filter-retry enforcement（advisory → filter-enforced）；(c) habitus 三维分类映射。
> **杠杆归位（配合 §2.2.5 阅读）**：Descriptive / Segmentation / population-alignment / nuwa / COLLEAGUE 均主要作用于 description；CADP 同时加入 inference-time filter-retry。当前内部强基线为 **Rich Cluster Narrative**，受 Scaling-Law 启发但不声称复现原论文。

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

### 3.2 Anti-patterns as Filter Triggers
- Goffman's Frame Analysis — frames 组织行为预期
- Anti-patterns = filter trigger，在 inference 阶段检测违规并触发 regeneration
- **Persona collapse 的机制分析**：RLHF 通过偏好优化压缩输出分布（"Alignment Tax"），使 agent 在冲突场景中趋向妥协/回避。描述性 persona 无法对抗这一倾向，因为身份标签不影响生成时的 token 分布偏移。
- Anti-patterns 作为 **filter-retry trigger**，通过 post-generation filter 拦截违规输出并 regeneration——这是 inference-time 干预，不碰权重；其有效性必须由 trigger audit 与 content-matched comparison 支持。
- "The Chameleon's Limit" 的实证发现为这一机制分析提供数据支撑：persona collapse 的速率与 RLHF 强度正相关。**但本文不声称 filter-retry 解决了权重级 collapse**——只声称在经验 fidelity 指标上优于 lever-1 方法
- **对 caricature 陷阱的防御论证（回应 §2.2.5 的 Scaling-Law / Promise-with-a-Catch）**：Chameleon's Limit §3.3 "fidelity breeds caricature" 显示高保真模型组间 Cohen's d>6，且 Promise-with-a-Catch 证明 LLM 生成内容越多偏差越大——两者都指向"杠杆 1（加 description）会恶化脸谱化"。CADP 的 filter-retry **不是**再注入 LLM 生成内容，而是**截断/约束**生成分布（虽受限于 inference-time 层面），理论预期可在**不增加 caricature** 的前提下提升 fidelity（§5.3 的 caricature 指标直接检验此预测）

---

## Chapter 4: Method — CADP Pipeline (~2 pages)
### 4.1 Overview
五步管线：Raw Corpus → Clustering → nuwa-skill Compilation → Agent Configuration → Sandbox Runtime

**方法学循环性防御声明**：聚类、编译、验证均基于同一社区数据，存在循环性质疑风险。本文通过三重机制防御：(1) 置换检验（§5.2 Condition 6 Shuffled）— 检测随机性；(2) 跨数据集迁移（§5.5 Wikipedia→Reddit）— 检测过拟合到特定社区；(3) 跨结构部分迁移（§5.5，共享分类器校准）— 在 community A 编译 .skill，部分迁移到结构不同的 community B（如 Reddit → GitHub：迁移可泛化组件 Expression DNA + 通用 Anti-patterns，含其校准的 θ_sem 语义阈值；平台特定的 Mind Models 与 Category C 行为分类器在 B 上重编译），若 fidelity 部分保持则说明捕获的是可泛化行为模式。~~(4) Held-out 事件预测准确率（§5.3 Predictive Fidelity layer）~~ — 已删除 2026-07-13（regex-based event detection 不可靠），未来替代方案应使用 CGA gold-standard `has_personal_attack` 标签。

### 4.2 Step 1: Clustering Typical Individuals

- **数据集**：English Wikipedia Talk Pages (WikiConv + CGA ConvoKit), 2001–2018 全量讨论页（约 80M utterances）
- **用户粒度**：以注册用户 / 稳定匿名 IP 为个体；跨年份累积行为特征，仅保留活跃度 floor 以上用户（最终 593,617 用户）
- **行为特征**：21 维行为向量，包括回复深度、回复率、编辑/删除/恢复率、毒性暴露、感叹率、疑问率、命名空间偏好、话题广度、 tenure、活动密度、burstiness CV 等（详见 `src/clustering/features.py:VECTOR_FIELD_NAMES`）
- **语言嵌入**：BGE  sentence embeddings；为每个用户聚合 toxicity-stratified 文本样本（高毒/低毒各保留一部分），确保语言 centroid 能覆盖冲突性情境
- **预处理**：`QuantileTransformer(output_distribution='normal')` 对行为特征做非参数正态化；对比 `RobustScaler(5-95)` 显著提升 silhouette 与 Davies-Bouldin
- **聚类**：K-Means, K=8（经 K=3–15 全量 sweep：silhouette、Davies-Bouldin、领域可解释性综合权衡）
  - K=8  silhouette = 0.259, Davies-Bouldin = 1.528
  - 聚类锁定结果：`outputs/stream_cache/clustering_k8_final_quantile.pkl`
- **K 选择标准**：silhouette score + Davies-Bouldin index + 领域可解释性；K=8 是 quality–interpretability 的折中，K>10 仅边际提升但可解释性下降
- **Skill 合并**：8 个聚类经可解释性审查后合并为 **6 个蒸馏 skill**（`outputs/skill_corpus_k8_quantile/wikiconv/cluster_map.json`）
  - cluster 1 → 0：Substantive discussant（深度讨论型）
  - cluster 5 → 4：Veteran generalist（资深泛化老手）
  - 独立保留：Niche terse specialist (2)、Confrontational editor (3)、Community patroller (6)、Expert fact-checker (7)
- **Cluster stability**：locked K8→6 bootstrap 已运行；final mean ARI=0.726、variance=0.021，未通过 mean≥0.80 且 variance≤0.02 的预注册阈值
- **下游语料**：每个 skill 抽取 30 个 centroid-最近代表用户，每人最多 300 条 utterances，生成约 15k typical utterances 用于 colleague/nuwa-skill 蒸馏

### 4.3 Step 2: nuwa-skill Compilation (5-Layer CADP Base)
- Top-N 最具代表性交互（N≈20 对话线程）
- 输出**5-layer .skill 文件**，结构继承自 **nuwa-skill 模板**（[nuwa-skill](https://github.com/alchaincyf/nuwa-skill)），应用到 WikiConv 聚类 archetype：
  - **Layer 1: Identity card** — archetype 名称 + 描述
  - **Layer 2: Mental models** — 推理模板、立场框架（5 个）
  - **Layer 3: Decision heuristics** — 决策规则（7 条）
  - **Layer 4: Expression DNA** — 允许的语言模式、词汇、句法
  - **Layer 5: Values and anti-patterns** — 显式 anti-patterns + 每 pattern 配 3 条真实 discuss 引用（用于 §4.4.1 trigger classifier 校准）+ Honest boundaries
- **为何 nuwa 而非 colleague 作 base**：
  - nuwa 三维（EDNA / MM / AP）一对一对映 CADP 设计
  - nuwa anti-patterns 结构化（named pattern + quotes）→ §4.4.1 Category B 语义 trigger 校准直接可用
  - colleague 6-layer 含 Work Skill 模块（scope/workflow/output preferences）与 social simulation 无关，Layer 5 "边界与雷区"是软 advisory 而非显式 anti-patterns
  - nuwa Honest boundaries 层与 §7.4 诚实声明对齐
- **colleague-skill 6-layer 蒸馏作为 methodology comparison baseline**（§5.2 条件 6），不消融、不裁剪，回答 "结构差异是否影响 fidelity"
- **Compilation 流程**（dual-pass，区别于 colleague 的 single-pass few-shot extraction）：
  - Pass 1 (positive case mining → Mental models + Decision heuristics + Expression DNA)
  - Pass 2 (negative case mining → Anti-patterns with structured trigger conditions，§4.4.1)
  - Pass 2 的 anti-pattern detection prompt 工程：从低 fidelity 交互中提取"该 archetype 不应做什么"，为每个 anti-pattern 生成结构化触发条件

### 4.4 Step 3: Three-Tier Filter-Retry Mechanism (Inference-Time Enforcement)
.skill 作为 **inference-time filter-retry 约束**（非 weights-level hard constraint），定义三层可度量执行机制：

> **范围声明**：本节三层机制全部在 prompt/filter-time 执行，**不访问 logits、不修改权重**；Exp1 只检验单模型 feasibility。

| Tier | 维度 | 执行方式 | 时机 |
|------|------|---------|------|
| Tier 1 | Expression DNA | Post-generation embedding max-z filter + regenerate | 生成后 |

> **Tier 1 校准说明（当前实验）**：每个 locked skill 从其 `typical.jsonl` 确定性抽样最多 400 条，80% 拟合 BGE-large centroid/std，20% 独立 validation 取 max-|z| 的 empirical q95。校准源 thread IDs 写入 hash 并从 evaluation 排除。Bonferroni 仅为无经验阈值时的 fallback；当前六类 q95 阈值约 4.04–5.04，validation observed false-reject rate 均为 5%。
| Tier 2 | Mind Models | Pre-generation retrieval-augmented context injection (dynamic rule selection) | 生成前 |
| Tier 3 | Anti-patterns | 当前仅保留未校准但高精度导向的 universal hostility lexical cues；命中后 block + constrained regeneration（最多 3 次）+ safe template | 生成后 |

> **Tier 2 与 Descriptive Persona system prompt 的技术区别**：Descriptive Persona 在 system prompt 中静态注入身份标签（一次性、不随对话上下文变化）。Tier 2 采用 **retrieval-augmented rule conditioning**：每轮根据当前对话状态（stance direction, conflict intensity, topic domain）从 Mind Models 规则库中动态检索最相关的 3-5 条推理模板注入 context。关键区别：(1) 动态 vs 静态——规则随上下文变化；(2) 条件化 vs 笼统——根据 agent 当前推理阶段选择匹配模板（如进入冲突阶段时检索"对抗性论证模板"而非"共识寻求模板"）；(3) Mind Models 包含从行为数据蒸馏的推理路径（如何从 A 推到 B），而非仅身份描述。若 §5.8 消融显示 minus Mind Models 下降不显著，则讨论 Tier 2 的角色可能主要为 Tier 1/3 提供 conditioning context（辅助功能），而非独立行为贡献。

#### 4.4.1 Trigger Formalization（新增 — Anti-pattern 执行的形式化）
当前 feasibility 不把自动生成的 trigger 当作已验证 classifier：
- polarity-ambiguous semantic phrases、全部 regex、Category-C action patterns 与非 hostility lexical cues 均 fail-closed 关闭；
- 仅保留与明确 hostile anti-pattern 关联的 conservative universal hostility keywords；运行 provenance 标记为 `conservative_universal_lexical_uncalibrated`；
- 因此当前结果**不能验证 archetype-specific Tier 3**，Tier 3 只作为 package 中的保守 guard。完整 Trigger Calibration Protocol 为 GO 后工作：
  - 标注协议：3 名标注者独立标注 500 条交互（per dataset），violation / non-violation 二分类
  - Inter-rater reliability：Fleiss' κ ≥ 0.6 后取多数票标签
  - 校准 / 验证分割：标注数据的 60% 用于 classifier 训练与阈值优化，40% 用于验证
  - 报告指标：Precision / Recall / F1；校准完成前不得启用 semantic/behavioral triggers
  - **目标性能**（v2 验证）：Precision ≥ 0.90（避免误伤正常行为），Recall ≥ 0.80（捕获大部分违规）
  - 跨域迁移测试：在 Wikipedia 校准的 trigger classifier 直接应用于 Reddit，报告迁移后的 Precision/Recall 下降程度（**v2 随 Cat C 一起 defer**）

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
- **所有模型统一基线执行方案**：output filtering + re-prompting（Tier 1 embedding filter + Tier 3 trigger block + regeneration）
- **当前不实现 logit bias intervention**：实验只使用 API 可用的 filtering + re-prompting。Chameleon's Limit 只作为限制背景，不被当作已证明的权重级因果结论。
  - **与现有工作的关系**：现有 constrained decoding 文献（JSON schema enforcement, grammar-guided decoding, safety refusal-and-retry 机制）处理的是**输出格式约束**或**安全过滤**。CADP 的 filter-retry 将约束对象从格式/安全扩展到**行为规则级别**（如论辩风格、交互模式约束），这一应用层面是新的，但底层机制（rejection sampling + re-prompting）与现有 alignment 技术有结构性联系
- **API 模型 (DeepSeek, GPT-4o, Claude)**：仅使用 output filtering + re-prompting（无法访问 logits）
- **开源模型 (Qwen / Llama)**：同上，v1 不实现 logit intervention

### 4.5 Step 4: Agent Configuration & Population

- **Feasibility 人口 = 18 agents = 6 locked skills × 3 agents**（`configs/exp1_v2.yaml`）。这是 balanced engineering population，不代表真实 population proportions。
- Engagement 从 cluster 内 empirical activity quantile 抽样并以非线性有界映射转为每轮参与率；允许低活跃 agent 跳过某轮。运行前必须报告 participation distribution，禁止退化为全员同一 ratio。
- .skill 通过三层 filter-retry 机制挂载为 inference-time 约束
- **Population 合理性检查**：
  - 18 agents 仅用于低成本方法 gate，不用于 confirmatory network inference
  - agent 与 round 是 run 内嵌套观测，不当作独立实验单位；当前 3 repeats 仅用于方向性 viability gate，不提供 80% confirmatory power

> **范围声明**：locked K8→6 仅作固定工程 partition。其 stability 未通过阈值，当前实验不把六类解释为稳定、自然存在的社区 archetypes。

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
- 当前 feasibility stage 为 8 轮/条件 × 3 paired repeats；长期 dynamics 与正式 confirmatory repeats 仅在 GO 后运行

### 4.7 Viability Conditions（与 `configs/exp1_v2.yaml` 同步）
**当前网格 4 条件 × 3 paired repeats × 8 rounds = 12 cells**：
1. **Descriptive Persona** — 标准静态描述基线
2. **Rich Cluster Narrative** — 外部 description diagnostic
3. **Nuwa Skill Advisory Only**（`cadp_advisory_nuwa`）— 与 Full 相同静态 skill 内容，无 retrieval/filter/retry；核心 control
4. **CADP Full (nuwa-distilled)** — 核心 treatment

双 distiller、minus Anti-patterns、Shuffled、三维消融与 α sensitivity 均为 **GO 后实验**，不属于当前 12-cell grid。所有四条件共享 locked clustering、人口分配、stimuli、evaluation manifest 和模型设置。
> Figure 2: CADP Pipeline 流程图 (含 nuwa 5-layer skill + three-tier filter-retry)

---

## Chapter 5: Experiment 1 — Method Validation (~3 pages)
### 5.1 Setup
- 数据集: **Wikipedia Talk Pages（primary）**。Reddit r/changemyview 迁移测试 deferred to v2 (§7.5)；GitHub Issues 移至 Exp 2 跨结构迁移节点，不入 Exp 1 主网格
- 模型: **1 模型 = DeepSeek-V4-Flash**。Cross-model generality deferred to future work (§7.5)；v1 单模型聚焦 method contribution 验证
- 总条件数: **4 conditions × 1 dataset × 1 model × 3 repeats = 12 simulation cells**
- 重复: 每条件 3 次；同一 dataset/model/repeat 的四条件共享 stimulus/evaluation manifest，按 repeat 配对
- **Agent 人口**：18 agents = 6 skills × 3 agents per skill（balanced feasibility population）
- **成本**：12 cells × 8 rounds × 18 agents；实际 planned turns 由共享 empirical-quantile engagement policy 决定
- **Power analysis（修正）**：paired two-sided test 在 d=0.5、α=0.05 下达到 80% power 约需 34 个独立 run pairs；n=3 的 power 极低，因此当前 gate 不做显著性推断，也不把 agent/round 当作独立样本。GO 后使用 pilot 的 run-level variance 决定 confirmatory repeats，或预先指定 hierarchical model
- **viability verdict**：Full 相对 advisory-only 在 action fidelity / interaction structure / independent linguistics 三个 family 中至少 2/3 达到 ≥5% improvement，且至少 2/3 paired repeats 同方向；任一 family 不得恶化 >10%。同时要求 message ratio≥0.95、safe-template≤0.10、action-text consistency≥0.90。
- **核心识别原则**：advisory 与 Full 使用同一 Nuwa skill、population、stimuli 与静态字段；当前 8-round gate 的差异为 retrieval/filter-retry。Reflection 在两臂均不触发，避免引入额外机制混淆。
- **训练/测试分离（fail closed）**：运行前从所有 `cluster_*/typical.jsonl` 读取 distillation `thread_id`，evaluation candidates 显式排除这些 IDs；manifest 写入 source-ID count/hash/overlap=0，加载时再次断言，任一 overlap 直接拒绝运行
- **统计定位**：当前 3-repeat 结果为方向性 Go/Stop evidence，不是 confirmatory significance evidence
- **可复现性 (Reproducibility)**：发布 (a) 完整 pipeline 代码 + colleague/nuwa-skill 编译器，(b) 化名（anonymized）聚合级 .skill 文件（不发布个体级，见 §7.5 dual-use），(c) 全部随机种子，(d) API 模型快照日期 + 开源模型 commit hash，(e) 标注协议 + 标注数据。数据集来自公开平台 API（CC-BY-SA / 平台 ToS 研究用途）

### 5.2 Viability Conditions

主网格严格等于 §4.7。预注册核心对比是 `cadp_full_nuwa` vs `cadp_advisory_nuwa`；Descriptive 与 Rich Narrative 只提供上下文，不参与机制 GO 判定。

**GO 后候选消融（不属于当前结果）**：
- CADP minus Expression DNA / minus Mental models / minus Decision heuristics（nuwa 三维解离检验，§5.8）
- COLLEAGUE capability-only（隔离 enforcement 增量）
- Clustering-Only Descriptive Persona（隔离聚类贡献）
- CADP Constraint-Only（mirror 消融）
- Pop-Aligned + CADP（叠加测试）
- Length-Matched Control（token-budget 隔离，DA-E1 反循环性）
- Segmentation Persona（复现 Qin et al. 2026）
- **colleague 版 minus Layer 5 Rejects + Correction**（colleague-specific 消融，对应 nuwa 条件 7）
- **colleague 版 Shuffled**（methodology robustness for permutation test）

> **Distiller 选择 rationale（2026-07-08）**：
> - nuwa 作 base 因 5-layer 一对一对映 CADP 三维；anti-patterns 层结构化（named pattern + 3 quotes）直接喂 §4.4.1 trigger classifier；Honest boundaries 层与 §7.4 诚实声明对齐
> - colleague 6-layer 含 Work Skill 模块（与 social sim 无关）+ Correction Log（与 filter-retry 平行，混淆 contribution），不适合作 base 但作 methodology head-to-head 回答结构问题
> - 所有消融/置换检验只在 nuwa 上跑（结构清晰可消融），colleague 版归附录 robustness
> **阶段化理由**：先用 12-cell viability gate 判断 CADP package 是否值得继续；只有 GO 后才投入机制识别、distiller robustness 与正式统计功效。

> **消融逻辑链**（逐层隔离结构贡献，主网格 + 附录合看）：
> - COLLEAGUE capability-only（附录）→ CADP minus Anti-patterns（主，nuwa）→ CADP Full（主，nuwa）：隔离 filter-retry 的增量
> - CADP Full (nuwa) vs CADP Full (colleague)（均主）：methodology comparison——distillation 结构是否影响 fidelity
> - Clustering-Only（附录）→ Descriptive Persona（主）：隔离聚类贡献
> - Length-Matched Control（附录）→ Descriptive Persona（主）→ CADP Full（主）：隔离 token 预算贡献（DA-E1 反循环性辩护）
> - CADP (Full) vs CADP (Shuffled)（均主，nuwa）：验证正确 .skill 分配的必要性
> - Pop-Aligned + CADP（附录）vs CADP alone（主）：测试属性/行为维度互补性
> 预期：Anti-patterns 移除对冲突/极化指标影响最大；COLLEAGUE capability-only 弱于 CADP minus Anti-patterns（说明 filter-retry 本身有价值）；CADP Full nuwa 与 colleague 差异取决于结构对齐度
> 关键对比预期：CADP (Full nuwa) vs Pop-Aligned — 属性分布匹配能接近但无法达到行为规则级保真度，差距在 Micro Behavior 层最显著

### 5.3 Four-Layer Evaluation Metrics
**GO/STOP 只使用三个 lower-is-better family distances**，避免把同一分布的多个变换当作独立票：`action_fidelity_distance = NED`；`interaction_structure_distance = cascade-length KS statistic`；`linguistic_fidelity_distance = mean(1 - discourse/sentiment/speech-act/SIP similarity)`。语言 family 使用 non-safe-template stratum；safe-template prevalence 由独立 guard 限制。Speech-act evaluation 固定使用本地模型，不让被测 Flash 模型评价自身。其余四层指标只作 diagnostics。`action_text_consistency` 是启发式完整性 guard，不是科学 outcome。
- **Macro Topology**: ΔQ Modularity, E-I Polarization Index, NED, **Coverage** (行为空间覆盖率, from Xiao et al. 2026)
  - Ground truth: 从真实社区同期交互日志中提取的网络结构（同一时间窗口的交互图）
- **Meso Dynamics**: Cascade Length Fit (KS-test), DTW, **Structural Fidelity** (交互网络结构相关, from Qin et al. 2026)
  - Ground truth: 真实社区中已发生争议事件的 cascade 长度分布、时间序列
- **Micro Behavior**: action marginal cosine（原字段名 `action_matrix_similarity`，仅 diagnostic）、RSA（diagnostic，已知易饱和）、**Uniformity Gap**、**Complexity Gap**
  - **Caricature Index + real reference**：分别计算 simulated/real role 间行为分布的标准化 centroid separation，报告 `caricature_index_sim`、`caricature_index_real`、绝对 gap，以及按 role 内 agent/user 重采样的 95% bootstrap CI。只有 sim-real gap 不增加时，才可支持“fidelity 提升但未加剧脸谱化”
  - Ground truth: 独立于 CADP 聚类的真实用户行为分布。**优先使用外部标注的行为类型分类**（Wu et al. 2025 audience segmentation 标签或人工标注的用户角色：moderator / provocateur / peacemaker / lurker）——`MetricsAggregator(role_labels_dir=...)` 检测 `data/role_labels/{dataset}.jsonl` 文件，存在时使用外部角色标签作为 ground truth，避免用 CADP 自己的聚类结果（循环依赖）。
  - **Fallback 路径（已实现）**：当外部标签文件不存在时，aggregator 退化为 Louvain 社区发现并记录 proxy provenance。当前 viability 的 NED/Uniformity/Complexity 不依赖 role labels；E-I 与 Caricature 强依赖标签语义，必须单独披露
- **Linguistics**: LSM (KL-divergence), SIP (Sentence-BERT cosine)
  - ⚠️ **Feature Leakage 注意**：Expression DNA 的蒸馏特征包含用词分布、句法模式。为避免自证循环，Linguistics 层评估使用**独立特征空间**——采用 Expression DNA 蒸馏时未使用的 NLP 特征（如 discourse marker 分布、sentiment trajectory shape、speech act ratio）作为评估特征，与蒸馏特征空间正交
- **Predictive Fidelity** — **已删除 2026-07-13**：原设计用仿真预测 held-out 真实交互结果（谁会冲突、说服是否成功、冲突是否升级），但 regex-based event detection 不可靠（context-blind keyword matching，precision/recall 均低），标注文件为 LLM 生成（Cohen's κ < 0.7）。CGA corpus 有 gold-standard `has_personal_attack` 标签，未来替代方案应直接使用。当前为 **4 层评估体系**（Macro / Meso / Micro / Linguistics）

### 5.3.5 Trigger Calibration Experiment（GO 后；不属于当前 12 cells）
- **目的**：独立评估 anti-pattern trigger classifier 的检测性能，作为 Three-Tier Filter-Retry 的前提验证
- **数据**：per dataset 500 条标注交互（3 名标注者，Fleiss' κ ≥ 0.6）
- **Split**：60% train / 40% test
- **未来报告**：
  - A 词法 + B 语义的 Precision / Recall / F1
  - 阈值 sensitivity（θ_sem ∈ {0.75, 0.80, 0.85, 0.90, 0.95}）
  - **Cat C 行为级 P/R/F1 deferred to v2**（infra 完整但 `data/trigger_calibration/` 标注数据未生产，详见 §4.4.1 deferred 声明）
- **跨数据集迁移性能（Wikipedia-trained → Reddit test）**：**v2 随 Cat C 一起 defer**（v1 单数据集 wiki-only）
- **通过标准（v2 验证目标）**：Precision ≥ 0.90, Recall ≥ 0.80（v1 不适用，因 Cat C 未训练；v1 仅报告 A/B 实测数字）

### 5.4 Cluster Stability Validation (新增)
- 多次重采样后计算 Adjusted Rand Index (ARI)
- 稳定判定预注册为 mean ARI ≥ 0.80 且 ARI variance ≤ 0.02；仅看低方差不足以证明稳定
- 配合 silhouette score + Davies-Bouldin index 选择 K
- 目的：证明聚类发现的行为类型是 robust pattern，非 artifact
- **Locked-vector bootstrap 协议**：从锁定 pickle 的 593,617 个有标签用户中，每轮有放回抽取 30,000 用户，重新拟合 QuantileTransformer(normal)+KMeans(K=8)，在固定 10,000-user evaluation sample 上计算 source-K8 ARI；随后用 bootstrap training contingency 的 Hungarian mapping 对齐 cluster IDs，应用预注册 1→0/5→4 merge，计算 final-K6 ARI。共 20 iterations
- **当前预实验结果**（`outputs/results/exp1_v2/clustering_stability_wikipedia.json`）：source-K8 mean ARI=0.772、variance=0.0098；final-K6 mean ARI=0.726、variance=0.0210、95% empirical interval [0.489, 0.956]，未通过 mean≥0.80 且 variance≤0.02 的稳定判定。该结果必须作为 clustering instability threat 报告，不能用“locked”替代稳定性证据
- **Locked-clustering 执行协议**：主实验仍加载 canonical pickle + merge map，以保证 skill IDs 一致；锁定解决复现性，但不被解释为稳定性证明
- **诚实披露**：skill 0 (Substantive discussant) 与 skill 4 (Veteran generalist) 的语言空间 centroid cosine = 0.740 > 0.70 阈值，存在语言相似性；二者区分依赖 behavioral axis（specialist deep-threading vs generalist bursty-poster），与 §4.2 behavior-first clustering 设计决策一致。skill 6 (Community patroller) silhouette = 0.161 为六 skill 中最低，其 emphatic / interpersonal-space 标签语义独立但行为信号偏弱

### 5.5 Cross-Dataset Transfer Test（GO 后）
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

### 5.6 Human Evaluation（GO 后）
- 3 名领域专家盲评 50 条仿真对话（v1 从原 100 缩减以控成本）
- 条件盲分配 (CADP vs Descriptive Persona vs Real)
- Cohen's κ ≥ 0.6 为可接受评分一致性
- 目的：外部效度补充，防止 metric overfitting

### 5.6.5 α Sensitivity Analysis（GO 后）
- **扫描设计**：三个 pairwise 5×5 sweep，每次固定第三个 tier 于 α=1.0：
  - α_expr × α_anti（fix α_mind=1.0）—— **primary**，回应 §3.2 anti-patterns 作为 filter trigger 的核心假设
  - α_expr × α_mind（fix α_anti=1.0）
  - α_mind × α_anti（fix α_expr=1.0）
  - 共 3 × 25 = **75 cells** per (dataset, model)。**v1 降级**：只在 Wikipedia × 单模型上跑（原 3 datasets × 4 models = 900 cells 不可行且边际收益低）。pairwise-at-1.0 覆盖三维 cube 在 (1,1,1) 角的三个 2D 面。**范围限制**：三维同时取中间值（如 (0.5,0.5,0.5)）的内点未探索，作为 acknowledged scope limit
- **报告**：
  - α_expr vs α_anti 的独立影响曲线（Mind Models 固定为 α=1.0）
  - 三个 pairwise sweep 各自的最优 cell + 等高线（plateau vs 尖锐 peak）
  - Wikipedia 上的最优 α 配置
  - robustness 检验：最优区域是否 plateau（`check_robustness` tolerance=0.05）而非尖锐 peak
- **目的**：证明 per-dimension α 的必要性（不同维度需要不同配置）并验证 filter-retry 强度的可控性

### 5.6.7 Persona Collapse Stress Test（GO 后）
- **目的**：纵向检验 CADP 的 filter-retry enforcement 是否能在长交互链中部分抵御 persona collapse——回答 Chameleon's Limit (arXiv:2604.24698) 提出的结构性失效问题（**注意**：filter-retry 不碰权重，预期是 partial mitigation 而非结构性解决，§7.4 acknowledged）
- **协议**：50+ 轮交互（高于主实验 §5.1 的 30-50 轮），vanilla Descriptive Persona vs CADP Full，per-turn 测量 silhouette / Davies-Bouldin / behavioral entropy / persona embedding drift
- **预期**：Descriptive Persona 在 20-30 轮后出现 collapse 信号（silhouette 单调下降，entropy 收缩）；CADP 保持 plateau
- 详细设计：`docs/r4_persona_collapse_stress_test.md`

### 5.7 Results
- 当前结果首先报告预注册 viability verdict（GO / CONDITIONAL_GO / STOP），不在 n=3 下使用“显著优于”措辞
- **Safe-template 分层报告**：所有 Linguistic 层指标按 `metadata.constraint_forced` 分层统计；当前 gate 同时要求 treatment safe-template rate ≤0.10。
- **Distiller methodology 对比**：仅在 viability GO 后追加，不混入当前 12 cells
- **Full vs Nuwa Skill Advisory Only（核心 viability comparison）**：按三个 metric families、最小效应与质量 guards 判断 enforcement 是否值得继续研究
- **Caricature 结果**：报告 sim、real、gap 与 bootstrap CI；若 CADP fidelity 改善但 caricature gap 增大，则 GO 结论必须附带风险标记
- 消融、distiller comparison、跨数据集、α sensitivity 与 human evaluation 均只在 GO 后追加，不能写入当前结果段。

> Table 1: viability 结果表 (4 conditions × 3 primary family distances + quality guards × 3 repeats)
> Table 2: 附录消融结果表 (附录条件 × 4 metric layers，Wikipedia 单模型 reduced grid) + COLLEAGUE capability-only 链式对比
> Table 2b: Clustering-Only vs Descriptive vs CADP（附录，聚类贡献隔离）
> Table 2c: Length-Matched Control vs Descriptive vs CADP（附录，token-budget 贡献隔离，DA-E1）
> Figure 3: paired-repeat slope/point plot — CADP Full vs Nuwa Skill Advisory Only（三个 metric families）
> Figure 4: 交互网络可视化对比

### 5.7.5 Self-Similarity Upper-Bound Analysis（GO 后）

**范围修正**：`real_history` 仅给出 metric pipeline 的 empirical self-similarity upper bound。它不是 weights-level solution，也不能量化 filter-retry 距权重级方法的距离。本节仅在 feasibility GO 后作为诊断分析运行。

**方法族定义**（`src/analysis/ceiling.py::DEFAULT_METHOD_FAMILIES`）：
- `none`：vanilla（无 persona）—— 地板参考
- `persona_prompting`：descriptive / segmentation / pop_aligned / clustering_only / length_matched_control / **rich_narrative** —— 杠杆 1（身份/属性/叙事注入，无 inference-time intervention）。rich_narrative 为此族的 ceiling（Scaling-Law 主张的"足够"方案）
- `distillation_advisory`：colleague_skill —— 规则蒸馏无 filter-retry
- `distillation_filter_enforced`：cadp_full_nuwa / cadp_full_colleague / cadp_shuffled / cadp_minus_* / cadp_constraint_only / pop_aligned_cadp —— 规则蒸馏 + 三层 filter-retry（杠杆 2 v1 实现，非 weights-level hard constraint）。nuwa 与 colleague 两 distiller 版本均归此族，用于 methodology comparison
- `self_similarity_reference`：real_history（§6.2 Exp2 replay arm）—— metric self-similarity upper bound

**计算**：对每个 (family, layer) cell，报告相对 self-similarity reference 的 observed remaining metric gap；不把该数解释为 weights-level ceiling gap。

**报告格式**（`format_ceiling_table`）：Markdown 表，行 = method family，列 = metric layer，cell = "remaining gap [95% CI] (best: <condition>)"。

**核心问题——本节必须回答**：
1. `distillation_filter_enforced`（CADP）相对 `persona_prompting`（含 rich_narrative ceiling）的边际 gap 闭合量是多少？
2. CADP 距离 empirical self-similarity upper bound 还剩多少 observed metric gap？该差距不能直接归因为 weights-level collapse。
3. 若 CADP 边际增益小（<10% additional gap closure），论文是否还成立？——成立，但 framing 转为 benchmark-style，强调"filter-retry 路线已饱和，需 weights-level 方法"作为 next-generation 方向

**可发表性保证**：本节的输出**不依赖 CADP 赢**。例如，即使 `distillation_filter_enforced` 在所有 layer 都只闭合 `persona_prompting` 已闭合 gap 的额外 10%，这本身是一个 finding："inference-time filter-retry 路线相对于 persona prompting 的边际贡献为 10%，剩余 Z% gap 需要 weights-level 干预（future work）"。Paper framing（method 主导 vs benchmark 主导）由 §5.1 framing pilot 决定，但本节内容在两种 framing 下都保留。

**与 §7.4 的关系**：Ceiling Analysis 的数字直接喂入 Threats to Validity 对方法学循环性、filter-retry 天花板、当前方法极限的讨论。本节是论文"诚实贡献"的核心——区别于过度营销的"hard enforcement"主张。

### 5.8 Analysis
- 各维度贡献分析 (基于消融)
- **三维 ↔ metric-layer 解离检验（验证 §3.1 预测）**：检验 minus Expression DNA / minus Mind Models / minus Anti-patterns 三者的 per-layer 损失是否解离——预测各自在不同 metric layer 达到峰值损失（EDNA→Linguistics / MM→Meso / AP→冲突极化）。报告三组 per-layer 损失的相关矩阵：若 off-diagonal 相关高（三维损失加载到单一因子）则如实承认分解解释力削弱；这是对 habitus 启发分解"非任意性"的直接检验
- **聚类贡献 vs 行为规则贡献的分解**：Clustering-Only 条件的性能定位了聚类的独立贡献上限；CADP Full - Clustering-Only = 行为规则蒸馏的净贡献
- **Tier 2 独立贡献检验**：CADP minus Mind Models 的下降幅度——若不显著，讨论 Tier 2 的 retrieval-augmented conditioning 是否主要为 Tier 1/3 提供 context（辅助角色）
- Descriptive Persona 在哪些指标上最接近 CADP
- **CADP vs Pop-Aligned 深入对比**：在哪些 metric layer 差距最大/最小？Pop-Aligned 在 Macro Topology（分布级）可能接近，但在 Micro Behavior（行为级）预期显著落后——量化"属性匹配 ≠ 行为匹配"
- **杠杆 1 vs CADP package 的边际增益**：当前 viability 只比较整体 package 与 Rich Cluster Narrative，不能独立识别 filter-retry 因果贡献；GO 后通过 content-only / enforcement controls 分解
- **Caricature 分析**：CADP 的 cluster 间 Cohen's d 随 fidelity 提升如何变化？若 fidelity 提升但 caricature 不增 → §3.2 "截断而非加内容" 论证成立；若 caricature 同步上升 → 如实承认 CADP 未逃脱 caricature 陷阱
- **COLLEAGUE → CADP 的增量来源（methodology comparison）**：CADP Full (nuwa 5-layer) vs CADP Full (colleague 6-layer) 在 4 metric layer 的差异定位结构贡献。若两者相近 → framework robust；若 nuwa 显著更优 → 5-layer 结构对 social sim 更适配
- **Pop-Aligned + CADP 叠加效果解读**：若叠加无显著增益，说明 CADP 的行为规则已隐含人口属性信息；若有增益，说明两维度正交互补
- **回应 arXiv:2501.08579**：当前单模型 viability 只能证明在一个模型配置上值得继续，不能确立一般性的 design fix；跨模型验证留待 GO 后
- Anti-patterns 作为 filter-retry trigger 的作用机制（**不主张 weights-level RLHF override**，§7.4 acknowledged）
- α Sensitivity: per-dimension 最优配置的维度差异性（附录 reduced grid）
- Predictive fidelity 已删除；当前 viability 由三个预注册 family distances 承担，Caricature gap 作为风险诊断，不使用 aggregate action cosine/RSA 作为 headline

---

## Chapter 6: GO-After Roadmap — Emergent Group Dynamics（不属于当前 feasibility paper）

> 本章仅保留为下一阶段设计备忘，不进入当前标题、摘要、贡献或结果叙事。当前 8-round、n=3 gate 不支持“恢复涌现动态”的结论；只有 feasibility GO 且增加独立 repeats 后才启动。

### 6.1 Motivation
若 Exp 1 达到预注册 GO，再检验 distilled-skill + filter-retry 是否改善 emergent dynamics；GO 之前不作方向性承诺。

### 6.2 Setup
- 从真实数据采样争议性场景
- CADP agents 交互 30-50 轮（**轮次对齐**：根据 pilot data 确定大多数争议性事件在真实社区中的交互轮次中位数，sandbox 轮次与之对齐；若真实事件中位数交互量为 M 轮，sandbox 设为 ⌈M × 1.5⌉ 以允许动态涌现空间）
- **对比条件（4 条件）**：真实历史 vs CADP (Full) vs Pop-Aligned Persona vs **CADP minus Anti-patterns**。Pop-Aligned 保留为最强竞品（Descriptive Persona 移至 §5 完整对比，Exp 2 聚焦最强 baseline）；**CADP minus Anti-patterns 是机制验证条件**——若移除 behavior track + filter-retry 后涌现动态（极化、冲突升级）显著退化甚至消失，则在动态轨迹层面验证 §3.1 的预测（Anti-patterns → 冲突/极化），把 Exp 2 的发现从相关性（"CADP 工作"）升级为机制性（"Anti-patterns filter-retry 驱动涌现"）
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
- CADP Full vs CADP minus Anti-patterns：behavior track + filter-retry 让 agent 群体"看见"了什么涌现动态（极化、冲突升级）、移除后又"看不见"了什么
- Scale test (robustness): N=30 vs N=100 fidelity 变化

> Figure 5: 极化指数时间演化曲线 (真实 vs CADP Full vs Pop-Aligned vs CADP minus Anti-patterns, 三平台)
> Figure 6: 模拟交互网络快照 (多时间点)
> Figure 7: 关键事件时机对比 (CADP vs real vs baseline)

### 6.7 Exploratory Findings
- 非预期的群体动态现象
- 在真实数据中验证对应

---

## Chapter 7: Discussion (~1 page)
### 7.1 Conditional Interpretation: When Does Inference-Time Filtering Help?
- RLHF 压缩行为多样性（Gao 2024 + Chameleon's Limit 一致诊断）
- 若 GO，讨论 distilled behavioral skill 与 filter-retry 可能通过何种路径改善 observed fidelity；若 STOP，讨论 prompt/filter-time 路线的局限
- **诚实声明**：filter-retry 不碰权重，无法结构性对抗权重级 RLHF attractor（Chameleon's Limit）。CADP 在经验 fidelity 上的提升 ≠ 解决 persona collapse 机制
- 三维协同必要性（§5.8 解离检验）
- **When does CADP help most**：基于 Ceiling Analysis（§5.7.5），CADP 增益集中在 Micro Behavior 层；在 Macro Topology 层可能与 lever-1 接近

### 7.2 Implications for Social Simulation Methodology
- 从"描述人是谁"到"蒸馏人怎么做"
- 对 ICWSM 社区的方法论建议

### 7.3 Relationship to Competing Methods
- vs COLLEAGUE.SKILL: CADP 不以 colleague 作 base structure（v1 reframe 2026-07-08）。CADP 借用 nuwa-skill 5-layer 模板；colleague 6-layer 作 methodology comparison baseline（条件 6）。差异：(a) 结构——nuwa 三维一对一对映 vs colleague 6-layer 含 Work Skill/Correction Log；(b) 应用域——social simulation vs 个人工具；(c) 执行机制——colleague advisory + Correction Log vs CADP filter-retry
- vs nuwa-skill: CADP 借用 nuwa 5-layer 结构模板作 base（Expression DNA / Mental models / Decision heuristics / Anti-patterns / Honest boundaries），但应用域不同——nuwa 蒸馏 public figure 认知框架，CADP 蒸馏社区 archetype 行为规则。CADP 增量：(a) nuwa 结构移植到 behavioral-trace clustering；(b) 三层 filter-retry enforcement；(c) habitus 三维分类；(d) social simulation 验证
- vs Population-Aligned Persona Generation: 不同层级（属性分布匹配 vs 行为规则蒸馏）；非替代关系，可叠加
- vs Restoring Heterogeneity / Simulation Boundary (Wu et al. 2025, arXiv:2506.19806): 补充而非替代（诊断 + 边界 vs 解决方案）
- vs Cognitive Heuristics (Zhu & Heydari 2026): 实证验证（理论推导 vs 可操作 pipeline）
- vs The Chameleon's Limit: 共享 persona-collapse 问题背景，但 CADP 只是 prompt/filter-time 候选干预，不是结构性或权重级解决方案。
- vs arXiv:2501.08579: CADP 是一个 design-side 候选干预；当前单模型 gate 不量化 model capability 与 design 的相对贡献
- vs **Scaling-Law (arXiv:2510.11734)** ⚠️: 该论文主张"更丰富 persona 就够了，无需 task-specific intervention"。CADP 的反驳 = filter-retry 是正交的杠杆 2，不是杠杆 1 的变体；§5.2 条件 4 直接对比，CADP 须在 Micro Behavior 层胜出才能立论
- vs **Promise-with-a-Catch (NeurIPS 2025)**: 该论文证明"LLM 生成内容越多偏差越大"。CADP 的回应 = filter-retry **不加** LLM 内容，而是**截断**生成分布（§3.2）；§5.3 Caricature Index 直接检验
- vs **PersonaEvolve / PEvo (arXiv:2509.16457)**: 该论文把显式行为指令当 failure mode（主张 implicit editing）。CADP 主张恰恰相反——inference-time filter-retry 是必要的；§5.2 结果判定孰是
- vs **PEP (arXiv:2603.03140)**: 两者均用 RAG/检索，但 PEP 检索描述性对话 persona（杠杆 1），CADP 检索可执行行为规则并通过 filter-retry 执行（杠杆 2）
- vs Park et al. (2024, arXiv:2411.10109, *preprint*, 1,052 individuals): interview-based vs behavioral-trace-based distillation；interview 依赖主动参与，CADP 可大规模从被动数据蒸馏
- vs "Habitus of GenAI" (Sage, 2025): 分析层面 vs 方法设计层面使用 habitus 概念

### 7.4 Threats to Validity（扩展为 Threats to Validity）
- **方法学循环性**：distillation 与 evaluation 来自同一平台，但 thread-level leakage 已通过 source-ID exclusion + manifest hash + overlap=0 fail-closed 防止。用户级与时间级分布依赖仍存在；Shuffled、跨数据集迁移与跨结构测试均为 GO 后验证
- **模型依赖**：当前 viability 只测试 DeepSeek-V4-Flash，不能声称跨模型一般性；更强或不同 alignment recipe 的模型可能改变 CADP 边际收益
- **与 LLM Simulation Boundary (arXiv:2506.19806) 的关系**：该论文提出 LLM 模拟的有效性边界。CADP 的适用边界：当社区行为高度依赖平台外知识（如线下社会关系、跨平台历史）时，仅从单平台行为轨迹蒸馏的 .skill 可能不足。
- **与 Population-Aligned Persona 的区分边界**：Pop-Aligned (arXiv:2509.10127) 做属性分布匹配，CADP 做行为规则蒸馏。两者非替代关系——§5.2 Condition 12 测试叠加效果（Pop-Aligned + CADP），实验结果将决定两者是互补还是冗余。
- **Logit steering 可行性限制**：行为规则级 logit intervention 与现有 constrained decoding 有结构性联系，应用层面（行为规则 vs 格式/安全）是新的。开源模型增强方案效果不确定，定位为 exploratory contribution。
- **Anti-pattern trigger 校准依赖**：trigger 阈值需 per-dataset 校准（§4.4.1），跨域迁移时 Precision/Recall 会下降——§5.5 报告具体下降程度。
- **Tier 3 当前范围**：没有 validated calibration report 时，runner fail-closed 地关闭 polarity-ambiguous semantic phrases、regex、非 hostility lexical cues 与 Category C；当前 viability treatment 实际为 Tier 1 + Tier 2 + `conservative_universal_lexical_uncalibrated` Tier 3。不得写成 archetype-specific Tier 3 已获验证。
- **Anti-patterns 编码社区偏见的风险**：CADP 从真实社区行为中蒸馏 anti-patterns，可能忠实地再现社区中的偏见性规范（如隐性歧视行为模式）。这是 "fidelity vs. ethics" 的结构性张力。**当前 release 不实现自动化 bias audit**（留作 future work，见 §7.5），所有 fidelity 数字均为 unaudited——即包含社区偏见的忠实再现，本文如实报告而非掩盖。
- **Bourdieu 框架的可证伪性**：本文不验证 habitus 社会学构念本身，但 §3.1 提出一个 *可证伪的设计预测*——三维消融应在不同 metric layer 上解离（而非加载到单一因子）。该预测在 §5.8 检验：若三组损失高度相关则如实承认分解解释力削弱。这是对"三维分解非任意"的检验，**不**构成对 Bourdieu 理论本身的验证。
- **训练级 vs inference-time 限制**：Chameleon's Limit 的 thinking/non-thinking 对比提示 extended reasoning 未必恢复 persona variation，但没有完成普遍 RLHF 因果识别。CADP 不触碰权重或 logits；即使 feasibility 为 GO，也只能报告 observed inference-time improvement，不能声称解决 persona collapse 或量化 weights-level gap。
- **Filter-retry 不是 hard constraint（2026-07-08 reframe）**：原 §1.4/§1.5/§2.2.5 曾主张 "generation-time hard behavioral enforcement"，经审查发现 v1 实现纯为 prompt/filter-time rejection sampling + regeneration，无法访问 logits 或权重。本文已统一降级为 "inference-time filter-retry"，并删除 logit intervention 主张（§4.4.4）。Reviewer 若发现残留 "hard constraint" 字眼，应视为遗漏，以本节为准。
- **Caricature 陷阱（2026-06-23 新增）**：Chameleon's Limit §3.3 "fidelity breeds caricature"（高保真 → Cohen's d>6）+ Promise-with-a-Catch（LLM 内容越多偏差越大）共同质疑"保真度提升必然改善真实感"。CADP 的论点是 enforcement 截断分布而非加内容，但此论点本身未经证实——§5.3 Caricature Index 检验：若 CADP 的 Cohen's d 反而更大，说明它只是更高效地脸谱化。
- **Habitus 三维映射的保真度（2026-06-23 新增）**：CADP 将 habitus 映射为"语言/认知/禁忌"三维，但社会学上 Bourdieu 的 habitus 通常分解为 dispositions(hexis) / tastes(aesthetics) / capital(resources)。CADP 的映射是**启发式**，非忠实社会学转译——本文将其作为 design blueprint（§3.1 已声明不验证构念本身），但需承认映射的正当性建立在 §5.8 消融解离的经验检验上，而非 Bourdieu 权威。
- **杠杆-2 新颖性的时效性（2026-06-23 新增，2026-07-08 措辞调整）**："inference-time filter-retry 将蒸馏行为规则注入 LLM 社会模拟"在文献检索（截至 2026-06）中无先例；极新 preprint 或非英文/非索引工作（尤其 nuwa-skill / COLLEAGUE 来源的中文圈工作）可能被遗漏。投稿前需复查。
- **聚类稳定性**：final-K6 bootstrap mean ARI=0.726 且未通过预注册稳定阈值；archetype 与下游结果可能对重采样敏感，GO 后需要 merge/K sensitivity 或 consensus clustering
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
| Table 1 | 4-condition viability grid × 5 pre-registered metrics + quality guards | 5.7 |
| Table 2 | Ablation + COLLEAGUE capability-only chain comparison (3 dimensions + capability-only baseline × 4 metric layers) | 5.7 |
| Table 2b | Clustering contribution isolation (Descriptive vs Clustering-Only vs CADP) | 5.7 |
| Table 2c | Token-budget contribution isolation — Length-Matched Control vs Descriptive vs CADP (DA-E1) | 5.7 |
| Table 2d | Ceiling Analysis — remaining sim-to-real gap per method family × metric layer (review-driven, ARS 2026-06-19) | 5.7.5 |
| Table 3 | Positioning comparison (5 methods × 7 dims) | 2.5 |
| Table 4 | Three-tier enforcement mechanism + trigger formalization | 4.4 |
| Table 5 | GO 后 trigger calibration results（当前 feasibility 不含） | 5.3.5 |
| Table 6 | α Sensitivity: 3 pairwise 5×5 sweeps × metric layers (75 cells, §5.6.5) | 5.6.5 |
| Figure 1 | Overview diagram | 1.5 |
| Figure 2 | CADP Pipeline flowchart (nuwa 5-layer skill + three-tier filter-retry) | 4.1 |
| Figure 3 | Paired-repeat Full vs Advisory comparison | 5.7 |
| Figure 4 | Interaction network visualization comparison | 5.7 |
| Figure 5 | Polarization index time evolution curves (real vs CADP vs baseline, 3 platforms) | 6.5 |
| Figure 6 | Simulated interaction network snapshots (multi-timepoint) | 6.6 |
| Figure 7 | Key event timing comparison (CADP vs real vs baseline) | 6.6 |
| Figure 8 | α Sensitivity heatmaps (per-dimension α × 4 metric layers, per dataset) | 5.6.5 |
| Figure 9 | Cross-structure transfer fidelity (Reddit→GitHub) | 5.5 |
