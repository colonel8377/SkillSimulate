# Paper Outline: Digital Habitus Distillation

## Metadata
- **Title**: Digital Habitus Distillation: From Persona Prompting to Behavioral Skills in LLM-based Social Simulation
- **Target Venue**: ICWSM / WWW (10-12 pages)
- **Format**: English, LaTeX, ACM double-column (WWW/`acmart` template)
- **Core Narrative**: Sim-to-Real Gap in LLM social simulation → Root cause: persona prompting carries identity labels not behavioral rules → CADP distills community behavioral traces into reusable `.skill` files → Distilled-skill agents recover emergent group dynamics (polarization, conflict, persuasion) that descriptive/pop-aligned/rich-narrative personas cannot → Inference-time filter-retry enforcement provides supporting mechanism with acknowledged ceiling
- **Headline Claim**: Distilled behavioral skills enable LLM agents to reproduce real community group dynamics that lever-1 persona methods cannot; filter-retry enforcement is a supporting mechanism, not a hard constraint at the weights level.
- **Theory Positioning**: Bourdieu's Habitus serves as a **design blueprint** (not a tested theoretical claim). The three-dimensional mapping provides the granularity for CADP's distillation axes; we do not measure "habitus scores" or validate the sociological construct itself. The blueprint yields **one falsifiable prediction** tested in §5.8: the three CADP ablations dissociate across metric layers (i.e., the decomposition is non-arbitrary rather than decorative).
- **Enforcement Honesty Caveat**: Chameleon's Limit (arXiv:2604.24698) establishes that persona collapse "resides in the weights, not the reasoning chain." CADP's three-tier mechanism operates entirely at prompt/filter-time (retrieval injection + post-generation filter-and-retry). It does **not** touch model weights or logits in v1. We therefore frame enforcement as **inference-time filtering**, not "hard behavioral constraint." §5.7.5 Ceiling Analysis and §7.4 quantify the distance between filter-retry and a weights-level ceiling.

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
- "Average Persona Problem" (Qin et al. 2026)
- Persona collapse 根因：RLHF 压缩行为多样性 + 描述性 prompt 缺乏行为约束 ("The Chameleon's Limit", arXiv:2604.24698)。该论文原文指出 collapse **"resides in the weights, not the reasoning chain"**——即 RLHF 优化几何产生 "Helpful Assistant" 吸引子驻留于权重，prompt-time 描述无法对抗
- 引用 Cognitive Heuristics 论证 (Zhu & Heydari 2026)

### 1.4 Our Approach: CADP — Distilled Behavioral Skills + Inference-Time Filtering
**核心论点（central thesis）**：缩小 sim-to-real gap 有两个正交杠杆——(1) 说什么（**description / persona 内容**）；（2）怎么约束生成（**inference-time intervention 机制**）。现有工作（descriptive → segmentation → pop-aligned → 更丰富叙事 persona → 迭代改写 persona）全部挤在杠杆 1 上；CADP 通过 **蒸馏出的三维行为规则 `.skill`** 同时拉动两个杠杆——skill 内容替代 description，inference-time filter-retry 替代无约束生成。
- 三维蒸馏（Expression DNA / Mind Models / Anti-patterns）为行为规则提供 granularity。**CADP 借用 nuwa-skill 的 5-layer 结构模板**（Expression DNA / Mental models / Decision heuristics / Anti-patterns / Honest boundaries）应用到社区聚类 archetype——不是用 nuwa 蒸馏 public figure，而是用 nuwa 的结构模板组织 WikiConv 行为轨迹。colleague-skill 6-layer persona（含 Work Skill + Correction Log）作为 methodology comparison baseline（§5.2 条件 6），不作为 CADP base structure
- **Inference-time 三层 filter-retry 执行**（pre-gen retrieval 注入 / post-gen embedding filter / post-gen trigger block + regeneration）是 CADP 的 supporting mechanism——**不是 weights-level hard constraint**。Chameleon's Limit 已证明 collapse 驻留权重，prompt/filter 时段干预有结构性天花板（§5.7.5 / §7.4 如实报告）
- **可证伪的 "so what" 预测**：distilled-skill agents 相对杠杆-1 强基线（含 Rich-Narrative persona）的增益，应集中在 held-out 事件预测（Predictive Fidelity）层与 emergent group dynamics（Exp 2 极化/冲突恢复）层

### 1.5 Contributions
1. **方法贡献（headline）**：首次将社区行为轨迹蒸馏为可复用 `.skill` 文件（三维规则：Expression DNA / Mind Models / Anti-patterns），并定义 **inference-time filter-retry enforcement**（三层：retrieval 注入 / embedding filter / trigger block+regeneration）作为补救 sim-to-real gap 的 supporting mechanism。CADP 借用 **nuwa-skill 5-layer 结构模板**应用到社区聚类 archetype（结构继承，非 novelty）；colleague-skill 6-layer persona 作为 methodology comparison baseline（条件 6），用于回答 "CADP framework 是否依赖特定 distillation 结构"。CADP 的真正创新是 **社区级行为蒸馏 + filter-retry 应用到 social simulation**——区别于 nuwa-skill 的 public-figure 应用域、colleague-skill 的个人工具定位，与 lever-1 persona 方法族
2. **实证贡献 (Exp 1)**：条件网格系统对比，**关键对比为 CADP distilled-skill（双 distiller: colleague + nuwa）相对杠杆-1 强基线（含 Rich-Narrative persona，复现 Scaling-Law arXiv:2510.11734）在 Predictive Fidelity 层的增益**（含 CADP minus Anti-patterns 机制消融 + Shuffled 置换检验 + distiller-robustness 对比 + Caricature 指标；8 主条件 + 附录消融）
3. **发现贡献 (Exp 2)**：distilled-skill agents 在三真实社区恢复 emergent group dynamics（polarization / conflict escalation / persuasion cascade）——回应 Gao et al. 2024 (NAACL Findings, peer-reviewed) 关于 LLM agent 向 consensus 漂移、无法涌现 polarization 的诊断
4. **理论贡献**：Bourdieu habitus 三维结构作为**设计蓝图**（非分析透镜），并检验其启发的三维分解是否经得起消融解离检验（§5.8）
5. **诚实贡献（§5.7.5 Ceiling Analysis）**：量化 inference-time filter-retry enforcement 距离 weights-level ceiling（perfect reference）的剩余 gap，不绑定 CADP 绝对排名——负结果也可发表

> Figure 1: 概览图

---

## Chapter 2: Related Work (~1.5 pages)
### 2.1 LLM-based Social Simulation
- Horton (2023), Park et al. (2023), Argyle et al. (2023)
- Park et al. (2024, *arXiv preprint* 2411.10109) — 大规模模拟 1,052 真实个体（interview-based persona），验证 generative agent 可信度；与 CADP 互补（interview-based vs behavioral-trace-based distillation）。**注**：截至 2026-06 仍为 arXiv 预印本（非 Nature 正式发表）；UIST 2023 的 25-agent "Generative Agents" 论文（Park et al. 2023）为不同工作
- Gap: 无人系统验证 persona vs 数据驱动方法的行为保真度

### 2.2 The Homogenization Problem
- **"LLM-Based Social Simulations Require a Boundary" (Wu et al. 2025, arXiv:2506.19806)** — 诊断平均人格问题并划定 LLM 社会模拟的有效性边界，无解决方案。**勘误**：outline 早期版本曾将 "Wu et al. 2025" 与 arXiv:2506.19806 当作两篇独立工作，实为同一篇（经 arXiv:2509.10127 参考文献表确认）
- Qin et al. (2026, arXiv:2604.06663) — audience segmentation，top-down 描述性方法。**勘误**：第一作者为 Qin，非早期版本的 "Li & Cheng"
- **"The Chameleon's Limit" (arXiv:2604.24698, Xiao et al., CMU/UChicago/MIT/RIKEN/JHU, 2026-04)** — 独立实证证据：**全部 10 个测试 LLM 均出现结构性 persona collapse**（BFI-44），归因于 RLHF 优化几何产生 "Helpful Assistant" 吸引子，且 collapse **"resides in the weights, not the reasoning chain"**。该论文仅提出诊断框架（Coverage/Uniformity/Complexity），把 "reward within-group behavioral variance 的训练目标" 明确列为未提出的 future work → **CADP 要补的"补救空间"真实存在且空缺**
- "What Limits LLM-based Human Simulation: LLMs or Our Design?" (arXiv:2501.08579) — 系统拆解 LLM 模拟偏差来源（模型能力 vs prompting 设计），结论指向设计缺陷为主因；本文进一步将"设计缺陷"定位到 persona prompting 的结构性局限并提出 CADP 作为改进方案
- Gap: segmentation 不保证个体级行为保真度；persona collapse 的根因（RLHF 压缩驻留权重 + 描述性 prompt 缺乏行为规则）未被解决；现有 gap 诊断缺乏可操作的结构性解决方案

### 2.2.5 Two Remediation Levers: Description vs Constraint（新增 — 核心定位框架）

本文用**两个正交杠杆**组织整个补救方法谱系，这是 CADP 定位的主轴：

- **杠杆 1 — Description（说什么）**：调整 persona 的内容丰富度。所有现有竞品都在此杠杆上：
  - Descriptive persona（身份/属性标签）
  - Segmentation（Qin et al. 2026，描述性分割标识符）
  - Population-Aligned Persona（arXiv:2509.10127，Importance Sampling + Optimal Transport 分布重采样）
  - **PersonaEvolve / PEvo（arXiv:2509.16457）** — 迭代改写 persona 的描述性特质（KL 向 SME 分布靠拢）；**该论文明确把"显式行为指令 / constraint injection"当成 failure mode**，与 CADP 的 filter-retry enforcement 理念正面冲突
  - **PEP — Persona Ecosystem Playground（arXiv:2603.03140）** — RAG 生成描述性对话 persona + 软 RQE 阈值修订
  - **"Scaling Law in LLM Simulated Personality"（arXiv:2510.11734）** ⚠️ **最强威胁**：随 persona 细节增加，到人类 Big Five 年龄曲线的欧氏距离 70.25→63.45→51.21→23.75 单调下降，主张 "more detailed persona profile is all you need，无需 task-specific intervention"。（注：仅基于单模型自报问卷，且被 Zierahn et al. 2026, arXiv:2603.19030 反驳）
  - **"LLM Generated Persona is a Promise with a Catch"（NeurIPS 2025 Position, arXiv:2503.16527）** — Meta Personas（无 LLM 生成内容）对齐最好；Descriptive/Generative Personas（LLM 内容越多）偏差越大，跨 6 个 LLM、3 个选举周期普适。**此结果动摇"内容越丰富越真实"的前提** → CADP 必须论证 enforcement ≠ 再加 LLM 内容，而是截断生成分布（见 §3.2）
- **杠杆 2 — Inference-Time Intervention（怎么约束生成）**：在生成前后注入/过滤行为规则。CADP 的实现是 **filter-retry enforcement**（retrieval-augmented rule injection + post-generation embedding/trigger filter + constrained regeneration）。**据文献检索（截至 2026-06），无竞品将蒸馏出的社区行为规则以 filter-retry 方式注入 LLM 社会模拟**——所有竞品（含 PEvo / PEP / Pop-Aligned / Scaling-Law）均通过描述性条件、分布重采样或迭代改写实现异质性，无一注入蒸馏出的、可执行的行为规则
- **重要诚实声明（2026-07-08 reframe）**：杠杆 2 在 v1 中以 **filter-retry** 实现，**非 weights-level hard constraint**。Chameleon's Limit (arXiv:2604.24698) 已证明 persona collapse "resides in the weights, not the reasoning chain"——filter-retry 无法结构性对抗权重级 RLHF attractor。本文如实报告此 ceiling（§5.7.5 / §7.4），headline 改为 **distilled behavioral skill + filter-retry 联合**，不再主张 "hard enforcement"

**Gap**：杠杆 1 已被充分探索甚至饱和（Scaling-Law 主张其已足够），杠杆 2（inference-time intervention）在 social simulation 域空缺。CADP 占据杠杆 2，并在 Predictive Fidelity 层验证"杠杆 1 不够、需 inference-time intervention"。**这要求 CADP 必须在实验中正面击败 Scaling-Law 的 Rich-Narrative 强基线（§5.2 条件 4），否则论点失败**（§5.1 framing pilot 的 kill condition）。**注意**：此 kill condition 检验的是 distilled-skill + filter-retry 是否优于 lever-1 ceiling，不主张 filter-retry 是 weights-level 解决方案

### 2.3 Population-Aligned Persona Generation（新增 — 最直接竞争者）

**方法概述**
- Population-Aligned Persona Generation (arXiv:2509.10127, Microsoft Research, 2025 *preprint*) — 用真实调查数据生成匹配人口**属性分布**的 persona 群体。生成目标是人口级 marginal alignment（per-attribute univariate 分布匹配），延续 Argyle et al. (2023) 的 silicon-samples 范式但扩展到 persona set 生成
- 同期相关工作：Survey-Derived Persona Prompt Collection (arXiv:2511.21722)、Deep Iterative Persona Alignment (preprints.org 2026) — 均属"分布匹配"族方法

**关键区别（CADP 的差异化 spine）**
- **层级区分**：Pop-Aligned 回答"有没有对的**类型**的人"（distributional validity, *types* of people present）；CADP 回答"这些人**做**得对不对"（behavioral fidelity, *behaviors* of those people）。前者是 *compositional* 维度，后者是 *interactional* 维度——两者正交，可叠加（§5.2 Condition 12 直接测试）
- **传递内容区分**：Pop-Aligned 传递属性标签（人口学 + 态度）；CADP 传递三维行为规则（Expression DNA / Mind Models / Anti-patterns）。属性标签是 *identity-level*，行为规则是 *rule-level*——前者对 RLHF 压缩无抵御能力，后者通过三层 filter-retry 在 inference 阶段截断违规输出
- **挂载机制区分**：Pop-Aligned 通过 system prompt 静态注入；CADP 通过 nuwa 5-layer .skill 文件 + three-tier filter-retry 动态执行
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
- COLLEAGUE.SKILL (Zhou et al. 2026, arXiv:2605.31264) — 面向个人工具，不声称行为保真度。**勘误**：经文献核实，COLLEAGUE.SKILL **本身是 dual-track**（capability track: practices / mental models / decision heuristics + bounded behavior track: communication style / interaction rules / correction history），早期版本"single-track"表述有误。**v1 reframe（2026-07-08）**：CADP 不再以 COLLEAGUE.SKILL 作 base structure。CADP 借用 **nuwa-skill 5-layer 模板**（Expression DNA / Mental models / Decision heuristics / Anti-patterns / Honest boundaries），因 nuwa 结构与 CADP 三维一对一对映、且 anti-patterns 层带结构化触发器（每 pattern 配 3 条真实引用，§4.4.1 trigger 校准直接可用）。COLLEAGUE 6-layer persona（含 Work Skill + Correction Log）作 **methodology comparison baseline**（§5.2 条件 6）——回答 "CADP framework 是否依赖特定 distillation 结构"。本文 baseline `colleague_skill_full` 为完整 6-layer colleague 蒸馏（capability + advisory behavior），**不消融、不裁剪**，作为 methodology head-to-head 对照
- nuwa-skill 框架（github.com/alchaincyf/nuwa-skill）— 5-layer cognitive OS 蒸馏（Identity / Mental models / Decision heuristics / Expression DNA / Anti-patterns + Honest boundaries），目标 = public figure perspective skill。**CADP 借用 nuwa 5-layer 模板**作 base structure（§4.3），应用到社区聚类 archetype。两者关系：结构继承 + 应用域扩展（public figure → community archetype）+ filter-retry enforcement + habitus 三维分类
- Zhu & Heydari (2026) — 理论推导，无实证
- Gap: 无工作将行为蒸馏 + inference-time filter-retry 应用于社会模拟并验证群体级保真度

### 2.5 Positioning Summary Table

| 维度 | Descriptive Persona | Segmentation | Pop-Aligned (2509.10127) | nuwa-skill | COLLEAGUE.SKILL | CADP (Ours) |
|------|--------------------|-------------|--------------------------|-----------|-----------------|-------------|
| 数据来源 | 人工/调查 | 人口学数据 | 调查数据 | 公开人物研究 | 个人行为轨迹 | 社区行为轨迹 |
| 传递内容 | 身份标签 | 人口学+心理标签 | 人口分布属性 | 5-layer 认知OS（含显式 anti-patterns）| 6-layer persona + Work Skill | 5-layer 行为规则（继承 nuwa）|
| 挂载方式 | system prompt | system prompt | system prompt | .skill 文件 | .skill 文件 | .skill filter-retry |
| 模拟目标 | 个体/群体 | 群体分布 | 群体分布匹配 | 个人/名人工具 | 个人工具 | 群体交互动态 |
| 约束机制 | 无 | 无 | 无 | advisory（Honest boundaries）| advisory + Correction Log | 三层 filter-retry (per-dim α) |
| RLHF override | ✗ | ✗ | ✗ | ✗ | 部分（negative examples + Correction）| 部分（filter-retry 不碰权重，§7.4 acknowledged） |
| 保真度验证 | 无 | 分布级 | 分布级 | 无 | 无 | 五层级系统验证 |

> **Table 说明**：✗/✓/部分 的区分旨在反映 continuum 而非 binary。CADP 的 base structure 继承自 nuwa-skill（5-layer 三维对映），colleague-skill 6-layer 作 methodology comparison baseline（条件 6）。CADP 相对两者的增量：(a) 社区级 behavioral-trace 应用域；(b) 三层 filter-retry enforcement（advisory → filter-enforced）；(c) habitus 三维分类映射。
> **杠杆归位（配合 §2.2.5 阅读）**：Descriptive / Segmentation / Pop-Aligned / nuwa / COLLEAGUE 均属杠杆 1（description）；CADP 同时拉动杠杆 2（inference-time filter-retry intervention）。**v1 reframe（2026-07-08）**：杠杆 2 以 filter-retry 实现，非 weights-level hard constraint（§1.4/§7.4）。表外补充一个杠杆-1 强基线 **Rich-Narrative persona（复现 Scaling-Law arXiv:2510.11734）** 作为 §5.2 条件 4，是 CADP 必须正面击败的对象。投稿前 Table 3 应增加 Rich-Narrative 行。

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
- Anti-patterns 作为 **filter-retry trigger**，通过 post-generation filter（Tier 1 embedding + Tier 3 trigger classifier）拦截违规输出并强制 regeneration——**这是 inference-time 干预，不碰权重**。"The Chameleon's Limit" 已证明 collapse 驻留权重；filter-retry 提供经验层面的 partial mitigation，结构性 ceiling 在 §5.7.5 / §7.4 如实报告
- "The Chameleon's Limit" 的实证发现为这一机制分析提供数据支撑：persona collapse 的速率与 RLHF 强度正相关。**但本文不声称 filter-retry 解决了权重级 collapse**——只声称在经验 fidelity 指标上优于 lever-1 方法
- **对 caricature 陷阱的防御论证（回应 §2.2.5 的 Scaling-Law / Promise-with-a-Catch）**：Chameleon's Limit §3.3 "fidelity breeds caricature" 显示高保真模型组间 Cohen's d>6，且 Promise-with-a-Catch 证明 LLM 生成内容越多偏差越大——两者都指向"杠杆 1（加 description）会恶化脸谱化"。CADP 的 filter-retry **不是**再注入 LLM 生成内容，而是**截断/约束**生成分布（虽受限于 inference-time 层面），理论预期可在**不增加 caricature** 的前提下提升 fidelity（§5.3 的 caricature 指标直接检验此预测）

---

## Chapter 4: Method — CADP Pipeline (~2 pages)
### 4.1 Overview
五步管线：Raw Corpus → Clustering → nuwa-skill Compilation → Agent Configuration → Sandbox Runtime

**方法学循环性防御声明**：聚类、编译、验证均基于同一社区数据，存在循环性质疑风险。本文通过四重机制防御：(1) 置换检验（§5.2 Condition 6 Shuffled）— 检测随机性；(2) 跨数据集迁移（§5.5 Wikipedia→Reddit）— 检测过拟合到特定社区；(3) Held-out 事件预测准确率（§5.3 Predictive Fidelity layer）— 检测方法是否捕获因果模式而非 spurious correlation；(4) 跨结构部分迁移（§5.5，共享分类器校准）— 在 community A 编译 .skill，部分迁移到结构不同的 community B（如 Reddit → GitHub：迁移可泛化组件 Expression DNA + 通用 Anti-patterns，含其校准的 θ_sem 语义阈值；平台特定的 Mind Models 与 Category C 行为分类器在 B 上重编译），若 fidelity 部分保持则说明捕获的是可泛化行为模式。Held-out 事件定义标准：由 2 名标注者独立编码争议性事件（conflict escalation / persuasion success / consensus formation），Cohen's κ ≥ 0.7 后取共识标签作为 ground truth。

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
- **Cluster stability**：重采样 ARI variance < 0.2（sweep 阶段已验证）
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

> **v1 reframe 重要声明（2026-07-08）**：本节三层机制全部在 prompt/filter-time 执行，**不访问 logits、不修改权重**。Chameleon's Limit 已证明 collapse 驻留权重，filter-retry 的天花板在 §5.7.5 / §7.4 如实报告。logit-level intervention 留作 future work（§4.4.4 deferred）。

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
- **所有模型统一基线执行方案**：output filtering + re-prompting（Tier 1 embedding filter + Tier 3 trigger block + regeneration）
- **v1 不实现 logit bias intervention**：原方案对开源模型施加 anti-pattern token sequence 负 logit bias。**2026-07-08 reframe 决策**：logit intervention 只能覆盖 §4.4.1 Category A（词法 token 级）trigger，无法覆盖 Category B（语义级 Sentence-BERT cosine）与 Category C（行为级 logistic regression）trigger——三者合计 anti-pattern 覆盖不全，且 Chameleon's Limit 已证明权重级问题非 prompt-time 能解。v1 全部走 filter-retry，logit intervention 列入 future work（§7.4 + §7.5）
  - **与现有工作的关系**：现有 constrained decoding 文献（JSON schema enforcement, grammar-guided decoding, safety refusal-and-retry 机制）处理的是**输出格式约束**或**安全过滤**。CADP 的 filter-retry 将约束对象从格式/安全扩展到**行为规则级别**（如论辩风格、交互模式约束），这一应用层面是新的，但底层机制（rejection sampling + re-prompting）与现有 alignment 技术有结构性联系
- **API 模型 (DeepSeek, GPT-4o, Claude)**：仅使用 output filtering + re-prompting（无法访问 logits）
- **开源模型 (Qwen / Llama)**：同上，v1 不实现 logit intervention

### 4.5 Step 4: Agent Configuration & Population

- **Agent 人口 = 6 个 skill × 每 skill 5 个 agents = 30 agents**（`exp1_full.yaml` population_size=30）
  - 保持 30 这个数字是为了与 outline §5.1 成本估算一致，同时确保每个 skill 有 5 个独立 agent 副本，避免单个 agent 的 idiosyncrasy 主导该 skill 的仿真
  - 比例按真实 cluster size 加权分配：skill 0 (25.3%), skill 2 (8.7%), skill 3 (7.0%), skill 4 (38.5%), skill 6 (10.9%), skill 7 (9.6%) → 30 agents 下四舍五入为：8 / 3 / 2 / 11 / 3 / 3
  - 若严格按 size 分配导致某 skill 只有 1–2 agents，则上取到 3，剩余从最大 skill 扣除；最终各 skill 至少 2 agents
- **Framing pilot 与 pilot**：使用 10 agents（`exp1_framing_pilot.yaml` / `exp1_pilot.yaml`），按最小可解释比例分配：2 / 1 / 1 / 3 / 1 / 2
- .skill 通过三层 filter-retry 机制挂载为 inference-time 约束
- **Population 合理性检查**：
  - Micro Behavior 层比较 agent-by-action matrix，30 agents 已能稳定估计动作分布
  - Meso/Macro 网络指标随 agent 数增长而改善，但 30 是成本-效益拐点（估计约 $2,000–4,000 全网格）
  - 5 repeats 下 30 agents 满足 §5.1 power analysis 对 Cohen's d ≥ 0.5 的 80% power 要求

> **与早期版本的改动**：早期 outline 写 "N≈30 agents" 但未说明与 K 的关系；K=8→6 skills 后，30 agents 明确为 6×5 结构，保证每个 skill 有独立统计样本。

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

### 4.7 Baselines Preview (v1 精简，详见 §5.2)
**主网格 8 条件**（每条 × 1 dataset × 1 model × 5 repeats = 40 runs）：
1. Vanilla LLM
2. Descriptive Persona
3. **Population-Aligned Persona** (复现 arXiv:2509.10127) — 最直接竞品
4. **Rich-Narrative Persona** (复现 Scaling-Law arXiv:2510.11734) — 杠杆-1 ceiling，核心论点存亡对照
5. **CADP Full (nuwa-distilled)** — 5-layer nuwa 结构 + 三层 filter-retry，**paper headline**
6. **CADP Full (colleague-distilled)** — 6-layer colleague 结构（完整，不裁剪），methodology comparison baseline
7. **CADP minus Anti-patterns (nuwa only)** — ablate 显式 anti-patterns 层，机制消融
8. **CADP Shuffled (nuwa only)** — 置换检验

**附录消融（reduced grid: Wikipedia × 单模型 × 5 repeats）**：Segmentation Persona、COLLEAGUE capability-only、Clustering-Only、CADP minus Expression DNA / Mind Models（nuwa）、CADP Constraint-Only、Pop-Aligned + CADP、Length-Matched Control、colleague 版 Shuffled（methodology robustness for permutation test）。完整定义见 §5.2。

> **精简说明（2026-07-08）**：原 13×3×4×5=780 cells 边际收益低且不可行。主网格 8 条件覆盖 lever-1 baseline 族 + CADP 主张（nuwa-primary + colleague comparison）+ 关键消融；附录保留全部原条件以 Wikipedia × 单模型 reduced grid 跑。Cell 总数从 780 降到 40 主 + ~50 附录。
> **Distiller 选择 rationale**：CADP Full 以 nuwa 为主条件，因 nuwa 5-layer 结构与 CADP 三维一对一对映、anti-patterns 层结构化可消融；colleague 6-layer 作 methodology head-to-head，回答"结构差异是否影响 fidelity"。所有消融/置换检验只在 nuwa 上跑（结构清晰），colleague 版归附录 robustness。
> **实验设计原则**：所有条件共享 Step 1 聚类结果（相同的 agent 分组结构），差异仅在于每组 agent 接收的 persona/skill 内容和 enforcement 机制。这隔离了"聚类结构"与"行为规则蒸馏"的贡献，避免混淆。
> Figure 2: CADP Pipeline 流程图 (含 nuwa 5-layer skill + three-tier filter-retry)

---

## Chapter 5: Experiment 1 — Method Validation (~3 pages)
### 5.1 Setup
- 数据集: **Wikipedia Talk Pages（primary）**。Reddit r/changemyview 迁移测试 deferred to v2 (§7.5)；GitHub Issues 移至 Exp 2 跨结构迁移节点，不入 Exp 1 主网格
- 模型: **1 模型 = DeepSeek-V4-Flash**。Cross-model generality deferred to future work (§7.5)；v1 单模型聚焦 method contribution 验证
- 总条件数: **8 主条件 × 1 dataset × 1 model = 8 cells × 5 repeats = 40 simulation runs**（v1 精简 2026-07-08；原 13×3×4×5=780 cells 不可行且边际收益低，附录保留扩展消融。**8 主条件 = 7 baseline/主条件 + CADP Full 双 distiller (colleague + nuwa)**，因两个 distiller 产出的 skill 内容不同，需都进主网格以回答 distiller-robustness 问题；消融条件只跑 colleague，nuwa 版本归附录）
- 重复: 每条件 5 次（framing-pilot effect-size dependent；d ≥ 0.5 → 5 repeats sufficient for 80% power at α=0.05）
- **Agent 人口**：30 agents = 6 skills × 5 agents per skill（outline §4.5）
- **每轮 API 调用数**：50 rounds × (30 agents + 环境/观察行为) ≈ 1,500 calls/run；40 runs 共约 60K calls（filter-retry 每条最多 3x tokens，估算上浮到 ~180K effective tokens）
- **样本量与可行性说明**：40 cells × 50 rounds × 30 agents per cell；API 成本估算（DeepSeek-V4-Flash @ $0.10/1K input + $0.40/1K output）约 $450–850。DeepSeek-V4-Pro 保留用于 skill compilation（已蒸馏完毕，不重复调用）。Cross-model analysis deferred to v2 (§7.5)
- **Power analysis**：基于 pilot data（Wikipedia 单数据集）的 effect size 估计，Cohen's d ≥ 0.5 时 5 次重复即可达到 80% power (α=0.05)；若 pilot 显示 d < 0.5，则增加至 10 次重复
- **Framing pilot（review-driven, ARS 2026-06-19）**：在主实验之前运行 `configs/exp1_framing_pilot.yaml`——**4 条件 (descriptive / pop_aligned / rich_narrative / cadp_full) × 10 repeats × Wikipedia × 单模型**，仅测 Micro Behavior + Predictive Fidelity 两层。**Pre-registered 决策规则**（unblinding 前冻结）：d ≥ 0.5 on ≥2 layers → method 主导 framing 可行；d ≈ 0.3 或单层 → benchmark 主导 framing；d < 0.3 → CADP 路线重构；CADP 输给 Pop-Aligned → benchmark framing + 重写 §2.3；**CADP 在 Predictive Fidelity 输给 Rich-Narrative（条件 4）→ 核心论点（inference-time intervention 必要性）失败，转 benchmark framing + 重写 §1.4/§2.2.5**。Framing pilot 与 §5.1 Power analysis 共享 effect-size 估计但作用不同
- **聚类共享原则**：所有 7 个主条件使用相同的 Step 1 聚类结果与 **6-skill 合并映射（`cluster_map.json`）**（相同 agent 分组），差异仅在 persona/skill 内容和 enforcement 机制。这隔离聚类结构贡献与行为规则蒸馏贡献
- **训练/测试分离披露（reframe v1, 2026-07-08 防止 train-on-test leakage）**：蒸馏训练集（`outputs/skill_corpus_k8_quantile/<dataset>/cluster_*/typical.jsonl`，每 cluster 的 representative threads，skill 文件基于这些 thread 蒸馏而成）与仿真评估集（raw WikiConv 样本，由 `WikipediaLoader` 在 `cfg.max_threads` cap 下采样）**不相交**。CADP conditions 在仿真阶段见到的是 held-out threads，而非 skill 蒸馏时见过的典型代表；这保证 kill-condition（CADP vs Rich-Narrative）对比的公平性——若 CADP 在 held-out threads 上仍胜出，则"行为规则 > 描述"的结论不受 train-on-test 质疑。所有 conditions 共用同一份 held-out 评估集
- **Statistical analysis（确认性 vs 探索性声明）**：Confirmatory comparisons = CADP Full vs 每个 baseline 在 5 个 metric layer 上的逐层检验，层内 Bonferroni 校正；报告 effect size (Cohen's d) + 95% CI，不仅 p 值。CADP minus Anti-patterns + Shuffled + 附录消融、α-sweep（§5.6.5）、迁移测试（§5.5）、trigger calibration（§5.3.5）归为 exploratory，描述性报告、不做 family-wise 校正，结论措辞相应弱化（"suggest"/"indicate" 而非 "prove"）
- **可复现性 (Reproducibility)**：发布 (a) 完整 pipeline 代码 + colleague/nuwa-skill 编译器，(b) 化名（anonymized）聚合级 .skill 文件（不发布个体级，见 §7.5 dual-use），(c) 全部随机种子，(d) API 模型快照日期 + 开源模型 commit hash，(e) 标注协议 + 标注数据。数据集来自公开平台 API（CC-BY-SA / 平台 ToS 研究用途）

### 5.2 Baselines (v1 精简为 8 主条件 + 附录消融，nuwa-primary，2026-07-08)

**主网格 8 条件**（每条 × 1 dataset × 1 model × 5 repeats）：
1. **Vanilla LLM** (无 persona) — 地板参考
2. **Descriptive Persona** (标准 system prompt) — lever-1 baseline
3. **Population-Aligned Persona** (复现 arXiv:2509.10127) — §2.3 最直接竞品
4. **Rich-Narrative Persona** (复现 Scaling-Law arXiv:2510.11734) — **杠杆-1 ceiling，CADP 核心论点存亡对照**：必须在 Predictive Fidelity 层显著超过 Rich-Narrative
5. **CADP Full (nuwa-distilled)** — **paper headline 主条件**。5-layer nuwa 结构（Identity / Mental models / Decision heuristics / Expression DNA / Values and anti-patterns）+ 三层 filter-retry。nuwa skill 蒸馏产出在 `data/nuwa_skills/`
6. **CADP Full (colleague-distilled)** — **methodology comparison baseline**。完整 6-layer colleague 结构（Layer 0-7 + Work Skill + Correction Log，不裁剪不消融）+ 三层 filter-retry。colleague skill 蒸馏产出在 `data/colleague_skills/`。回答 "nuwa 5-layer vs colleague 6-layer 结构在 social simulation 上谁更 fidelity"。若 colleague ≈ nuwa → framework robust 到结构；若 nuwa > colleague → 5-layer 结构本身贡献（paper finding）
7. **CADP minus Anti-patterns (nuwa only)** — 机制消融：ablate nuwa 第 5 层 "Values and anti-patterns"（保留 Honest boundaries）。验证 anti-patterns filter-retry 的增量贡献。colleague 版（ablate Layer 5 Rejects + Correction）归附录
8. **CADP Shuffled (nuwa only)** — 置换检验：保持 6-skill agent 分组结构不变，随机重分配 .skill 到错误 skill。colleague 版归附录

**附录消融（reduced grid：Wikipedia × 单模型 × 5 repeats）**：
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
> **精简理由**：原 13×3×4×5=780 cells 边际收益低。8 主条件覆盖 lever-1 baseline 族 + CADP 主张（nuwa-primary + colleague comparison）+ 关键消融/置换（nuwa）。Cell 总数 780 → 40 主 + ~50 附录。**此条件需新增代码**：新 agent adapter for Rich-Narrative，经 `src/agents/registry.py` 注册

> **消融逻辑链**（逐层隔离结构贡献，主网格 + 附录合看）：
> - COLLEAGUE capability-only（附录）→ CADP minus Anti-patterns（主，nuwa）→ CADP Full（主，nuwa）：隔离 filter-retry 的增量
> - CADP Full (nuwa) vs CADP Full (colleague)（均主）：methodology comparison——distillation 结构是否影响 fidelity
> - Clustering-Only（附录）→ Descriptive Persona（主）：隔离聚类贡献
> - Length-Matched Control（附录）→ Descriptive Persona（主）→ CADP Full（主）：隔离 token 预算贡献（DA-E1 反循环性辩护）
> - CADP (Full) vs CADP (Shuffled)（均主，nuwa）：验证正确 .skill 分配的必要性
> - Pop-Aligned + CADP（附录）vs CADP alone（主）：测试属性/行为维度互补性
> 预期：Anti-patterns 移除对冲突/极化指标影响最大；COLLEAGUE capability-only 弱于 CADP minus Anti-patterns（说明 filter-retry 本身有价值）；CADP Full nuwa 与 colleague 差异取决于结构对齐度
> 关键对比预期：CADP (Full nuwa) vs Pop-Aligned — 属性分布匹配能接近但无法达到行为规则级保真度，差距在 Micro Behavior 和 Predictive Fidelity 层最显著

### 5.3 Five-Layer Evaluation Metrics
- **Macro Topology**: ΔQ Modularity, E-I Polarization Index, NED, **Coverage** (行为空间覆盖率, from Xiao et al. 2026)
  - Ground truth: 从真实社区同期交互日志中提取的网络结构（同一时间窗口的交互图）
- **Meso Dynamics**: Cascade Length Fit (KS-test), DTW, **Structural Fidelity** (交互网络结构相关, from Qin et al. 2026)
  - Ground truth: 真实社区中已发生争议事件的 cascade 长度分布、时间序列
- **Micro Behavior**: Action Matrix Similarity (Frobenius), RSA, **Uniformity** (行为分布熵, 检测同质化), **Complexity** (跨agent行为方差)
  - **Caricature Index（新增，回应 Chameleon's Limit §3.3 + Promise-with-a-Catch）**：cluster 间行为分布的 Cohen's d（组间区分度）。Chameleon's Limit 报告高保真模型组间 Cohen's d>6（"fidelity breeds caricature"）；本指标直接检验 CADP 是否在**不增加 caricature** 的前提下提升 fidelity（验证 §3.2 "enforcement 截断分布而非加内容" 的论证）。**此指标需在 `src/evaluation/aggregator.py` 新增 `_compute_caricature`**。预期：CADP 的 Caricature Index 不显著高于 baseline，甚至更低（enforcement 约束而非脸谱化）；若 CADP 的 Cohen's d 反而显著更大 → 它只是更高效地制造脸谱化，核心卖点受挫
  - Ground truth: 独立于 CADP 聚类的真实用户行为分布。**优先使用外部标注的行为类型分类**（Wu et al. 2025 audience segmentation 标签或人工标注的用户角色：moderator / provocateur / peacemaker / lurker）——`MetricsAggregator(role_labels_dir=...)` 检测 `data/role_labels/{dataset}.jsonl` 文件，存在时使用外部角色标签作为 ground truth，避免用 CADP 自己的聚类结果（循环依赖）。
  - **Fallback 路径（已实现）**：当外部标签文件不存在时，aggregator 退化为 Louvain 社区发现（从真实交互图推断）作为 ground-truth proxy，并记录到 `datasets_using_role_label_proxy`，供 §7.4 报告 per-dataset validity 差异。Louvain proxy 仍然独立于 CADP 的聚类（不存在循环依赖），但弱于外部角色标签——这是一个 acknowledged limitation 而非 spec 违反。Micro 层的 Frobenius / RSA / Uniformity / Complexity 度量直接比较 agent-by-action matrix，不强依赖 cluster 标签，因此 proxy fallback 的影响主要在 Macro E-I polarization 层
- **Linguistics**: LSM (KL-divergence), SIP (Sentence-BERT cosine)
  - ⚠️ **Feature Leakage 注意**：Expression DNA 的蒸馏特征包含用词分布、句法模式。为避免自证循环，Linguistics 层评估使用**独立特征空间**——采用 Expression DNA 蒸馏时未使用的 NLP 特征（如 discourse marker 分布、sentiment trajectory shape、speech act ratio）作为评估特征，与蒸馏特征空间正交
- **Predictive Fidelity** (新增第五层, from Qin et al. 2026):
  - 用仿真预测 held-out 真实交互结果 (谁会冲突、说服是否成功、冲突是否升级)
  - Ground truth 编码协议：2 名标注者独立编码 held-out 事件结果，Cohen's κ ≥ 0.7 后取共识标签
  - 这是最强的 "so what" 指标 — 仿真能否预测未见过的事件

### 5.3.5 Trigger Calibration Experiment (新增 — §4.4.1 的独立验证)
- **目的**：独立评估 anti-pattern trigger classifier 的检测性能，作为 Three-Tier Filter-Retry 的前提验证
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
- **Locked-clustering 协议（reframe v1, 2026-07-08）**：主实验不再每 cell 现场重新聚类，而是加载 **canonical 锁定聚类 pickle（`outputs/stream_cache/clustering_k8_final_quantile.pkl`）+ 6-skill 合并映射**。锁定聚类的 cluster IDs (0,2,3,4,6,7) 与 distilled skill 文件一一对应；现场重新聚类会产生不同 IDs → CADP conditions 静默加载错误 skill。锁定聚类质量 = silhouette 0.259, Davies-Bouldin 1.528, n=593,617 users（来自 `outputs/skill_corpus_k8_quantile/wikiconv/quality_report.json`），K=8→6 合并由 K-sweep 在锁定阶段选定，不再做 live ARI bootstrap（live bootstrap 只对未锁定聚类有意义）
- **诚实披露**：skill 0 (Substantive discussant) 与 skill 4 (Veteran generalist) 的语言空间 centroid cosine = 0.740 > 0.70 阈值，存在语言相似性；二者区分依赖 behavioral axis（specialist deep-threading vs generalist bursty-poster），与 §4.2 behavior-first clustering 设计决策一致。skill 6 (Community patroller) silhouette = 0.161 为六 skill 中最低，其 emphatic / interpersonal-space 标签语义独立但行为信号偏弱

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
- 3 名领域专家盲评 50 条仿真对话（v1 从原 100 缩减以控成本）
- 条件盲分配 (CADP vs Descriptive Persona vs Real)
- Cohen's κ ≥ 0.6 为可接受评分一致性
- 目的：外部效度补充，防止 metric overfitting

### 5.6.5 α Sensitivity Analysis (附录 reduced grid，2026-07-08 降级)
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

### 5.6.7 Persona Collapse Stress Test (新增 — 直接回应 "The Chameleon's Limit")
- **目的**：纵向检验 CADP 的 filter-retry enforcement 是否能在长交互链中部分抵御 persona collapse——回答 Chameleon's Limit (arXiv:2604.24698) 提出的结构性失效问题（**注意**：filter-retry 不碰权重，预期是 partial mitigation 而非结构性解决，§7.4 acknowledged）
- **协议**：50+ 轮交互（高于主实验 §5.1 的 30-50 轮），vanilla Descriptive Persona vs CADP Full，per-turn 测量 silhouette / Davies-Bouldin / behavioral entropy / persona embedding drift
- **预期**：Descriptive Persona 在 20-30 轮后出现 collapse 信号（silhouette 单调下降，entropy 收缩）；CADP 保持 plateau
- 详细设计：`docs/r4_persona_collapse_stress_test.md`

### 5.7 Results
- CADP (Full, both distillers) 全面显著优于所有 baseline (8 主条件对比)
- **Safe-template 分层报告**：所有 Linguistic 层指标按 `metadata.constraint_forced` 分层统计——safe-template 输出（filter-retry fallback，§4.4.2 step 4）单独报告，不混入主指标均值。这对 `cadp_minus_anti_patterns`（条件 7）尤其关键：该条件 filter-retry 关闭，对比 fair 需明确分层
- **Distiller methodology 对比（条件 5 vs 6）**：CADP Full nuwa (5-layer) vs colleague (6-layer) 在 5 metric layer 的差异。若两者表现相近（Cohen's d < 0.3）→ CADP framework robust 到 distillation 结构；若 nuwa 显著更优 → 5-layer 结构对 social simulation 更适配（paper finding，§5.8 深入分析）
- **COLLEAGUE capability-only vs CADP 链式对比**（附录 reduced grid）：COLLEAGUE capability-only < CADP minus Anti-patterns filter-retry < CADP Full，逐层隔离 filter-retry 的结构贡献
- **Clustering-Only vs Descriptive Persona vs CADP**（附录）：隔离聚类贡献——若 Clustering-Only 显著优于 Descriptive Persona，需报告聚类占 CADP 优势的比例
- 消融分析: 各维度独立贡献量化（附录 reduced grid）
- 关键对比: CADP vs Pop-Aligned 逐维度差异；CADP vs Descriptive 逐维度差异
- **CADP vs Rich-Narrative（核心论点验证，§1.4/§2.2.5）**：在 Predictive Fidelity 层 CADP 是否显著超过 Scaling-Law 的"更丰富叙事"基线——这是 inference-time intervention 必要性的直接证据。预期：差距集中在 Predictive Fidelity + Micro Behavior，Macro Topology 可能接近
- **Caricature Index 结果**：CADP 的 cluster 间 Cohen's d 是否显著低于/不高于杠杆-1 方法——验证"filter-retry 截断分布而非脸谱化"
- **Pop-Aligned + CADP 叠加效果**（附录）：是否显著优于纯 CADP（属性/行为互补性）
- 置换检验: Shuffled（agent→skill 重新分配）显著弱于 Full
- **Length-Matched Control 结果**（附录）：CADP / Descriptive Persona 是否显著优于等长随机描述（DA-E1 反循环性辩护）
- 跨数据集迁移: Wikipedia→Reddit 全组件迁移 vs 方法论迁移结果（§5.5）
- 预测性保真度: held-out 事件预测准确率
- Trigger calibration: per-category P/R/F1 + 跨域迁移性能（§5.3.5）
- α Sensitivity: per-dimension α 曲线（附录 reduced grid）
- 跨模型一致性（deferred to v2）
- **Human evaluation 结果**：3 名专家盲评（Cohen's κ ≥ 0.6）中 CADP 的辨识度——专家能否区分 CADP 仿真 vs 真实交互（作为外部效度的核心证据，在主结果中报告）

> Table 1: 主结果表 (8 conditions × 5 metric layers × 1 model × 1 dataset)
> Table 2: 附录消融结果表 (附录条件 × 5 metric layers，Wikipedia 单模型 reduced grid) + COLLEAGUE capability-only 链式对比
> Table 2b: Clustering-Only vs Descriptive vs CADP（附录，聚类贡献隔离）
> Table 2c: Length-Matched Control vs Descriptive vs CADP（附录，token-budget 贡献隔离，DA-E1）
> Figure 3: 雷达图 — 8 条件 5 层指标对比
> Figure 4: 交互网络可视化对比

### 5.7.5 Ceiling Analysis（**论文核心诚实锚点 — v1 升格 2026-07-08**）

**目的**：负结果保险 + 诚实承认 filter-retry 天花板。无论 CADP 在 7 条件对比中排第几，本节都贡献一个可发表的 finding——按 method family × metric layer 报告"剩余 sim-to-real gap"，量化当前方法族（**特别是 inference-time filter-retry**）在闭合 gap 上的天花板。回应 panel 共识 + Chameleon's Limit 的根本质疑：contribution 不应绑死在 CADP 绝对排名上，且必须诚实报告 filter-retry 距 weights-level 解决方案还差多远。

**方法族定义**（`src/analysis/ceiling.py::DEFAULT_METHOD_FAMILIES`）：
- `none`：vanilla（无 persona）—— 地板参考
- `persona_prompting`：descriptive / segmentation / pop_aligned / clustering_only / length_matched_control / **rich_narrative** —— 杠杆 1（身份/属性/叙事注入，无 inference-time intervention）。rich_narrative 为此族的 ceiling（Scaling-Law 主张的"足够"方案）
- `distillation_advisory`：colleague_skill —— 规则蒸馏无 filter-retry
- `distillation_filter_enforced`：cadp_full_nuwa / cadp_full_colleague / cadp_shuffled / cadp_minus_* / cadp_constraint_only / pop_aligned_cadp —— 规则蒸馏 + 三层 filter-retry（杠杆 2 v1 实现，非 weights-level hard constraint）。nuwa 与 colleague 两 distiller 版本均归此族，用于 methodology comparison
- `perfect_reference`：real_history（§6.2 Exp2 replay arm）—— 自相似 ceiling，定义 "zero gap"

**计算**：对每个 (family, layer) cell，取该 family 在该 layer 上的最佳 condition 的归一化 fidelity，与 `perfect_reference` 的 fidelity 之差即 "remaining gap"。Per-metric direction（higher-better vs lower-better）按 `DEFAULT_METRIC_DIRECTION` 处理。Bootstrap 95% CI（n_resamples=1000）。

**报告格式**（`format_ceiling_table`）：Markdown 表，行 = method family，列 = metric layer，cell = "remaining gap [95% CI] (best: <condition>)"。

**核心问题——本节必须回答**：
1. `distillation_filter_enforced`（CADP）相对 `persona_prompting`（含 rich_narrative ceiling）的边际 gap 闭合量是多少？
2. CADP 距离 `perfect_reference`（zero gap）还剩多少？剩余 gap 的源头是什么（§7.4 归因到 weights-level collapse / 平台特定规则 / 数据 sparsity）？
3. 若 CADP 边际增益小（<10% additional gap closure），论文是否还成立？——成立，但 framing 转为 benchmark-style，强调"filter-retry 路线已饱和，需 weights-level 方法"作为 next-generation 方向

**可发表性保证**：本节的输出**不依赖 CADP 赢**。例如，即使 `distillation_filter_enforced` 在所有 layer 都只闭合 `persona_prompting` 已闭合 gap 的额外 10%，这本身是一个 finding："inference-time filter-retry 路线相对于 persona prompting 的边际贡献为 10%，剩余 Z% gap 需要 weights-level 干预（future work）"。Paper framing（method 主导 vs benchmark 主导）由 §5.1 framing pilot 决定，但本节内容在两种 framing 下都保留。

**与 §7.4 的关系**：Ceiling Analysis 的数字直接喂入 Threats to Validity 对方法学循环性、filter-retry 天花板、当前方法极限的讨论。本节是论文"诚实贡献"的核心——区别于过度营销的"hard enforcement"主张。

### 5.8 Analysis
- 各维度贡献分析 (基于消融)
- **三维 ↔ metric-layer 解离检验（验证 §3.1 预测）**：检验 minus Expression DNA / minus Mind Models / minus Anti-patterns 三者的 per-layer 损失是否解离——预测各自在不同 metric layer 达到峰值损失（EDNA→Linguistics / MM→Meso / AP→冲突极化）。报告三组 per-layer 损失的相关矩阵：若 off-diagonal 相关高（三维损失加载到单一因子）则如实承认分解解释力削弱；这是对 habitus 启发分解"非任意性"的直接检验
- **聚类贡献 vs 行为规则贡献的分解**：Clustering-Only 条件的性能定位了聚类的独立贡献上限；CADP Full - Clustering-Only = 行为规则蒸馏的净贡献
- **Tier 2 独立贡献检验**：CADP minus Mind Models 的下降幅度——若不显著，讨论 Tier 2 的 retrieval-augmented conditioning 是否主要为 Tier 1/3 提供 context（辅助角色）
- Descriptive Persona 在哪些指标上最接近 CADP
- **CADP vs Pop-Aligned 深入对比**：在哪些 metric layer 差距最大/最小？Pop-Aligned 在 Macro Topology（分布级）可能接近，但在 Micro Behavior 和 Predictive Fidelity（行为级）预期显著落后——量化"属性匹配 ≠ 行为匹配"
- **杠杆 1 vs 杠杆 2 的边际增益分解（§1.4/§2.2.5 核心分析）**：沿杠杆 1 从 descriptive → segmentation → pop_aligned → rich_narrative 的 fidelity 增益曲线是否已饱和（plateau）？CADP（杠杆 2）相对 rich_narrative（杠杆 1 ceiling）的增益集中在哪一层？若杠杆 1 已 plateau 而 CADP 仍能提升 → 证明杠杆 2 是新方向；若杠杆 1 仍在上升 → CADP 须论证 enforcement 带来 description 无法触及的增益（如 Predictive Fidelity）
- **Caricature 分析**：CADP 的 cluster 间 Cohen's d 随 fidelity 提升如何变化？若 fidelity 提升但 caricature 不增 → §3.2 "截断而非加内容" 论证成立；若 caricature 同步上升 → 如实承认 CADP 未逃脱 caricature 陷阱
- **COLLEAGUE → CADP 的增量来源（methodology comparison）**：CADP Full (nuwa 5-layer) vs CADP Full (colleague 6-layer) 在 5 metric layer 的差异定位结构贡献。若两者相近 → framework robust；若 nuwa 显著更优 → 5-layer 结构对 social sim 更适配
- **Pop-Aligned + CADP 叠加效果解读**：若叠加无显著增益，说明 CADP 的行为规则已隐含人口属性信息；若有增益，说明两维度正交互补
- **回应 "What Limits LLM Simulation" (arXiv:2501.08579)**：跨模型对比分析 deferred to v2 (§7.5)；v1 单模型结果已足以验证 method contribution——若 CADP 在单模型上显著优于 lever-1 baseline，则 design fix 效果确立；跨模型 generality 留后续验证
- Anti-patterns 作为 filter-retry trigger 的作用机制（**不主张 weights-level RLHF override**，§7.4 acknowledged）
- α Sensitivity: per-dimension 最优配置的维度差异性（附录 reduced grid）
- Predictive fidelity 的 "so what" 论证

---

## Chapter 6: Experiment 2 — Social Simulation: Recovering Emergent Group Dynamics (~2.5 pages)

> **v1 emphasis 升格（2026-07-08）**：Exp 2 是论文核心实证贡献——distilled-skill agents 是否能恢复 lever-1 方法无法涌现的群体动态（polarization / conflict escalation / persuasion cascade）。Gao et al. 2024 (NAACL Findings, peer-reviewed) 已证明 LLM agent 向 consensus 漂移、无法涌现 polarization；若 CADP-simulated agents 恢复 polarization 模式，这是 Q1 级 finding（lit review Theme E 直接背书）。Exp 1 验证方法可信度，Exp 2 给出"so what"答案。

### 6.1 Motivation
Exp 1 证明 distilled-skill + filter-retry 在 metrics 层可信 → Exp 2 验证：distilled-skill agents 能否恢复 lever-1 方法失败的 emergent group dynamics？回应 Gao 2024 的 consensus-drift 诊断。

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
### 7.1 Why Distilled Skills Outperform Persona: Inference-Time Filtering + Behavioral Rules
- RLHF 压缩行为多样性（Gao 2024 + Chameleon's Limit 一致诊断）
- Distilled behavioral skill 提供具体规则（三维），filter-retry 在 inference 阶段截断违规生成
- **诚实声明**：filter-retry 不碰权重，无法结构性对抗权重级 RLHF attractor（Chameleon's Limit）。CADP 在经验 fidelity 上的提升 ≠ 解决 persona collapse 机制
- 三维协同必要性（§5.8 解离检验）
- **When does CADP help most**：基于 Ceiling Analysis（§5.7.5），CADP 增益集中在 Predictive Fidelity + Micro Behavior 层；在 Macro Topology 层可能与 lever-1 接近

### 7.2 Implications for Social Simulation Methodology
- 从"描述人是谁"到"蒸馏人怎么做"
- 对 ICWSM 社区的方法论建议

### 7.3 Relationship to Competing Methods
- vs COLLEAGUE.SKILL: CADP 不以 colleague 作 base structure（v1 reframe 2026-07-08）。CADP 借用 nuwa-skill 5-layer 模板；colleague 6-layer 作 methodology comparison baseline（条件 6）。差异：(a) 结构——nuwa 三维一对一对映 vs colleague 6-layer 含 Work Skill/Correction Log；(b) 应用域——social simulation vs 个人工具；(c) 执行机制——colleague advisory + Correction Log vs CADP filter-retry
- vs nuwa-skill: CADP 借用 nuwa 5-layer 结构模板作 base（Expression DNA / Mental models / Decision heuristics / Anti-patterns / Honest boundaries），但应用域不同——nuwa 蒸馏 public figure 认知框架，CADP 蒸馏社区 archetype 行为规则。CADP 增量：(a) nuwa 结构移植到 behavioral-trace clustering；(b) 三层 filter-retry enforcement；(c) habitus 三维分类；(d) social simulation 验证
- vs Population-Aligned Persona Generation: 不同层级（属性分布匹配 vs 行为规则蒸馏）；非替代关系，可叠加
- vs Restoring Heterogeneity / Simulation Boundary (Wu et al. 2025, arXiv:2506.19806): 补充而非替代（诊断 + 边界 vs 解决方案）
- vs Cognitive Heuristics (Zhu & Heydari 2026): 实证验证（理论推导 vs 可操作 pipeline）
- vs The Chameleon's Limit: 共同问题诊断，CADP 提供结构性解决方案；但需注意该论文指出 collapse "resides in the weights"——CADP 的 prompt/filter-time enforcement 能否对抗权重级吸引子是核心经验风险（base-model 对照 §5.1 检验）
- vs "What Limits LLM Simulation" (arXiv:2501.08579): CADP 是该论文提出问题（LLMs or Design?）的 design-side 回答；跨模型分析（§5.8）量化 design fix 相对于 model capability 的边际贡献
- vs **Scaling-Law (arXiv:2510.11734)** ⚠️: 该论文主张"更丰富 persona 就够了，无需 task-specific intervention"。CADP 的反驳 = filter-retry 是正交的杠杆 2，不是杠杆 1 的变体；§5.2 条件 4 直接对比，CADP 须在 Predictive Fidelity 层胜出才能立论
- vs **Promise-with-a-Catch (NeurIPS 2025)**: 该论文证明"LLM 生成内容越多偏差越大"。CADP 的回应 = filter-retry **不加** LLM 内容，而是**截断**生成分布（§3.2）；§5.3 Caricature Index 直接检验
- vs **PersonaEvolve / PEvo (arXiv:2509.16457)**: 该论文把显式行为指令当 failure mode（主张 implicit editing）。CADP 主张恰恰相反——inference-time filter-retry 是必要的；§5.2 结果判定孰是
- vs **PEP (arXiv:2603.03140)**: 两者均用 RAG/检索，但 PEP 检索描述性对话 persona（杠杆 1），CADP 检索可执行行为规则并通过 filter-retry 执行（杠杆 2）
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
- **权重级 vs 推理链级 collapse（核心经验风险，2026-07-08 reframe 升格）**：Chameleon's Limit (arXiv:2604.24698) 原文指出 persona collapse "resides in the weights, not the reasoning chain"。**CADP v1 的三层 filter-retry 全部是 prompt/filter-time 干预，不触碰权重或 logits**（§4.4.4 logit intervention 已删除）——若 RLHF 吸引子确实驻留权重，filter-retry 的经验提升可能是部分 mitigation 而非结构性解决方案。§5.7.5 Ceiling Analysis 量化此 gap：distillation_filter_enforced 相对 perfect_reference 的 remaining gap 即 filter-retry 路线的天花板。**本文如实报告此 ceiling，不主张 "解决" persona collapse**——只主张在经验 fidelity 上优于 lever-1 方法族。Weights-level 干预（logit bias / fine-tuning / activation steering）留作 future work。
- **Filter-retry 不是 hard constraint（2026-07-08 reframe）**：原 §1.4/§1.5/§2.2.5 曾主张 "generation-time hard behavioral enforcement"，经审查发现 v1 实现纯为 prompt/filter-time rejection sampling + regeneration，无法访问 logits 或权重。本文已统一降级为 "inference-time filter-retry"，并删除 logit intervention 主张（§4.4.4）。Reviewer 若发现残留 "hard constraint" 字眼，应视为遗漏，以本节为准。
- **Caricature 陷阱（2026-06-23 新增）**：Chameleon's Limit §3.3 "fidelity breeds caricature"（高保真 → Cohen's d>6）+ Promise-with-a-Catch（LLM 内容越多偏差越大）共同质疑"保真度提升必然改善真实感"。CADP 的论点是 enforcement 截断分布而非加内容，但此论点本身未经证实——§5.3 Caricature Index 检验：若 CADP 的 Cohen's d 反而更大，说明它只是更高效地脸谱化。
- **Habitus 三维映射的保真度（2026-06-23 新增）**：CADP 将 habitus 映射为"语言/认知/禁忌"三维，但社会学上 Bourdieu 的 habitus 通常分解为 dispositions(hexis) / tastes(aesthetics) / capital(resources)。CADP 的映射是**启发式**，非忠实社会学转译——本文将其作为 design blueprint（§3.1 已声明不验证构念本身），但需承认映射的正当性建立在 §5.8 消融解离的经验检验上，而非 Bourdieu 权威。
- **杠杆-2 新颖性的时效性（2026-06-23 新增，2026-07-08 措辞调整）**："inference-time filter-retry 将蒸馏行为规则注入 LLM 社会模拟"在文献检索（截至 2026-06）中无先例；极新 preprint 或非英文/非索引工作（尤其 nuwa-skill / COLLEAGUE 来源的中文圈工作）可能被遗漏。投稿前需复查。
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
| Table 2 | Ablation + COLLEAGUE capability-only chain comparison (3 dimensions + capability-only baseline × 5 metric layers) | 5.7 |
| Table 2b | Clustering contribution isolation (Descriptive vs Clustering-Only vs CADP) | 5.7 |
| Table 2c | Token-budget contribution isolation — Length-Matched Control vs Descriptive vs CADP (DA-E1) | 5.7 |
| Table 2d | Ceiling Analysis — remaining sim-to-real gap per method family × metric layer (review-driven, ARS 2026-06-19) | 5.7.5 |
| Table 3 | Positioning comparison (5 methods × 7 dims) | 2.5 |
| Table 4 | Three-tier enforcement mechanism + trigger formalization | 4.4 |
| Table 5 | Trigger calibration results (P/R/F1 per category × 3 datasets) | 5.3.5 |
| Table 6 | α Sensitivity: 3 pairwise 5×5 sweeps × metric layers (75 cells, §5.6.5) | 5.6.5 |
| Figure 1 | Overview diagram | 1.5 |
| Figure 2 | CADP Pipeline flowchart (nuwa 5-layer skill + three-tier filter-retry) | 4.1 |
| Figure 3 | Radar chart — 13 conditions × 5 metric layers | 5.7 |
| Figure 4 | Interaction network visualization comparison | 5.7 |
| Figure 5 | Polarization index time evolution curves (real vs CADP vs baseline, 3 platforms) | 6.5 |
| Figure 6 | Simulated interaction network snapshots (multi-timepoint) | 6.6 |
| Figure 7 | Key event timing comparison (CADP vs real vs baseline) | 6.6 |
| Figure 8 | α Sensitivity heatmaps (per-dimension α × 5 metric layers, per dataset) | 5.6.5 |
| Figure 9 | Cross-structure transfer fidelity (Reddit→GitHub) | 5.5 |
