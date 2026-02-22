# Efficient Agents: Building Effective Agents While Reducing Cost

> **Source:** OPPO AI Agent Team (arXiv:2508.02694v1, August 6, 2025)
> **Code:** [github.com/OPPO-PersonalAI/OAgents](https://github.com/OPPO-PersonalAI/OAgents)

---

## Abstract

The remarkable capabilities of Large Language Model (LLM)-driven agents have enabled sophisticated systems to tackle complex, multi-step tasks, but their escalating costs threaten scalability and accessibility. This work presents the first systematic study of the efficiency-effectiveness trade-off in modern agent systems, addressing the critical need for cost-effective designs without sacrificing performance.

Three key research questions are investigated:
1. How much complexity do agentic tasks inherently require?
2. When do additional modules yield diminishing returns?
3. How much efficiency can be gained through the design of efficient agent frameworks?

Through an empirical analysis on the GAIA benchmark, the authors evaluate the impact of LLM backbone selection, agent framework designs, and test-time scaling strategies. Using the **cost-of-pass** metric, they quantify the efficiency-performance trade-off across these dimensions.

**Key Result:** Efficient Agents retains 96.7% of the performance of OWL (a leading open-source agent framework), while reducing operational costs from $0.398 to $0.228, resulting in a 28.4% improvement in cost-of-pass.

---

## 1. Introduction

Agent research has reached an inflection point similar to the efficiency turn in NLP. While increasingly sophisticated agent architectures can solve remarkably complex problems, their costs scale prohibitively. Industry deployments reveal this tension starkly: cutting-edge agent products (e.g., DeepResearch, Manus) demonstrate impressive capabilities but suffer from exorbitant operating costs due to explosive LLM call overhead. Some systems require hundreds of API calls per task, rendering them economically unsustainable despite their technical brilliance.

### Contributions

- Thorough analysis of factors causing significant economic overhead in generic LLM-based agent systems
- Proposal of Efficient Agents, an optimized framework achieving 96.7% of OWL's performance while reducing costs by 28.4%

---

## 2. Preliminaries

### 2.1 Setup

Factors evaluated on the GAIA benchmark include:
- **Backbone LLM** selection
- **Agent framework** design (planning mechanisms, tool usage, memory module)
- **Test-time scaling** strategies

### 2.2 Metrics

The **cost-of-pass** metric represents the expected monetary cost of using a model to generate a correct solution for a problem:

```
v(m, p) = C_m(p) / R_m(p)
```

Where:
- `C_m(p)` = cost of a single inference attempt = `n_in(m,p) * c_in(m) + n_out(m,p) * c_out(m)`
- `R_m(p)` = success rate (proportion of correct responses)

---

## 3. Efficiency-Performance Trade-off Analysis

### 3.1 Backbones

| Method | Cost-of-Pass (all) | Accuracy (all) | Cost/$ (all) | Tokens (all) |
|---|---|---|---|---|
| GPT-4.1 | 0.98 | 53.33% | 0.705 | 243K |
| Claude 3.7 Sonnet | 3.54 | 61.82% | 2.190 | 680K |
| Qwen3-235B-A22B | 0.22 | 27.27% | 0.040 | 72K |
| Qwen3-30B-A3B | 0.13 | 17.58% | 0.023 | 65K |
| QwQ-32B | 0.23 | 22.42% | 0.120 | 142K |
| o1 | 3.66 | 52.12% | 1.908 | 69K |

**Key Findings:**
- Claude 3.7 Sonnet achieves the highest accuracy (61.82%) but its cost-of-pass is significantly higher (3.54 vs. 0.98 for GPT-4.1)
- Sparse models like Qwen3-30B-A3B exhibit superior efficiency (0.13 cost-of-pass) despite modest accuracy (17.58%)
- As task difficulty increases from Level 1 to Level 3, cost-of-pass rises dramatically across large reasoning models

> **Finding:** As task difficulty escalates, cost-of-pass of reasoning models dramatically increases and efficiency significantly deteriorates, posing a formidable challenge for deploying these models in intricate agentic environments.

### 3.2 Test-time Scaling Strategies

Best-of-N (BoN) performance using a Progress Reward Model (PRM) implemented via GPT-4o:

| N | Cost-of-Pass (all) | Accuracy (all) | Cost/$ (all) | Tokens (all) |
|---|---|---|---|---|
| 1 | 0.98 | 53.33% | 0.521 | 243K |
| 2 | 1.17 | 54.55% | 0.639 | 298K |
| 4 | 1.28 | 53.94% | 0.691 | 325K |

> **Finding:** The marginal performance gains of BoN come at a disproportionate computational cost, highlighting the need for more efficient test-time scaling strategies in an agent setting.

### 3.3 Planning

Planning uses a ReAct-style approach where the agent generates an explicit plan, follows it step by step, and periodically revises.

| Max Steps | Plan Interval | Cost-of-Pass (all) | Accuracy (all) |
|---|---|---|---|
| 12 | 1 | 0.98 | 53.33% |
| 8 | 1 | 0.70 | 52.73% |
| 4 | 1 | 0.48 | 41.82% |
| 12 | 2 | 1.04 | 57.58% |
| 12 | 4 | 1.01 | 53.33% |

**Key Findings:**
- Increasing max steps from 4 to 8 significantly improves accuracy (58.49% to 69.81%) but also increases cost-of-pass
- Beyond a threshold, further increasing max steps does not enhance performance but continues to increase costs

> **Finding:** Current models struggle with reasoning length regulation, often exhibiting overthinking that inflates costs when problems are insoluble. Moderate planning complexity significantly enhances efficiency.

### 3.4 Tool Using

| Source | Tool | Search Num | Cost-of-Pass (all) | Accuracy (all) |
|---|---|---|---|---|
| Simple | Crawler | 10 | 1.32 | 53.33% |
| Multi | Crawler | 10 | 0.81 | 59.39% |
| Simple | Browser-Complex | 10 | 0.88 | 49.09% |
| Simple | Browser-Simple | 10 | 1.59 | 54.18% |
| Simple | Crawler | 5 | 1.17 | 53.33% |
| Simple | Crawler | 3 | 1.31 | 49.09% |

> **Finding:** Varying tool configurations, such as increasing search sources, simplifying browser operations, and expanding reformulated queries for web searching, demonstrably enhance both effectiveness and efficiency in information retrieval.

### 3.5 Memory

Six memory configurations were tested:

| Memory | Cost-of-Pass (all) | Accuracy (all) | Cost/$ (all) | Tokens (all) |
|---|---|---|---|---|
| Simple | 0.74 | 56.36% | 0.419 | 194K |
| Summarized | 1.52 | 51.52% | 0.782 | 367K |
| w/o Extra | 0.98 | 53.33% | 0.521 | 243K |
| Extra Summarized | 1.08 | 52.73% | 0.567 | 236K |
| Extra Fixed | 1.04 | 53.94% | 0.561 | 240K |
| Extra Hybrid | 1.29 | 54.55% | 0.703 | 259K |

Memory configurations defined:
- **Simple Memory:** Only historical observations and actions kept in context window
- **Summarized Memory:** All information summarized by an LLM and embedded in a vector database
- **w/o Extra Memory:** Full history of every step kept in context with no extra memory
- **Extra Summarized/Fixed/Hybrid Memory:** Additional memory layers augmenting step history

> **Finding:** The Simple Memory design, retaining only the agent's observations and actions, is sufficient to achieve both effectiveness and efficiency.

### 3.6 Holistic Analysis

The choice of **backbone** exerts the most significant impact on overall system performance. The **maximum number of steps** and **tool usage** also play critical roles. In contrast, BoN and memory mechanisms have negligible effects on effectiveness, though redundant designs can increase computational costs.

---

## 4. Efficient Agents: Tricks of the Trade

### Optimal Configuration

| Component | Backbone | Max Step | Plan Interval | Search Source | Search Num | BoN | Memory |
|---|---|---|---|---|---|---|---|
| **Settings** | GPT-4.1 | 8 | 1 | Multi | 5 | 1 | Simple |

### Comparison with Other Frameworks

| Agent | Cost-of-Pass (all) | Accuracy (all) | Cost/$ (all) | Tokens (all) |
|---|---|---|---|---|
| OWL | 0.75 | 53.33% | 0.398 | 189K |
| SmolAgents | 5.82 | 53.33% | 3.104 | 146K |
| **Efficient Agents** | **0.55** | **51.52%** | **0.285** | **127K** |

Efficient Agents achieves a cost reduction of 28.4% while maintaining comparable performance (96.7% of OWL).

---

## 5. Related Work

### 5.1 LLM-driven Agents
Notable systems include OpenAI's Deep Research (67.36% on GAIA) and OWL (69.7% on GAIA), demonstrating the significant potential of LLM-based agent systems for intricate tasks requiring sophisticated reasoning, planning, and tool utilization.

### 5.2 Efficient NLP
Strategies from efficient NLP (knowledge distillation, token budgets, communication pruning) inform the design of cost-effective agent architectures. Approaches like Token-Budget-Aware LLM Reasoning, AgentPrune, and BudgetMLAgent explore tiered model architectures combining lower-cost and high-performance LLMs.

---

## 6. Conclusion

This paper provides:
1. A comprehensive analysis of architectural choices contributing to economic overhead in contemporary agent systems
2. Efficient Agents, a framework engineered for optimal balance between task performance and computational cost

Efficient Agents dynamically adapts its complexity to the demands of the task at hand, achieving 96.7% of OWL's performance while drastically reducing operational costs, resulting in a 28.4% improvement in cost-of-pass.

---

## Appendix

### Default Experimental Setup

| Component | Backbone | Max Step | Plan Interval | Search Source | Search Num | BoN | Memory |
|---|---|---|---|---|---|---|---|
| **Settings** | GPT-4.1 | 12 | 1 | Simple | 10 | 1 | Simple |

### Prompts

**Memory Prompt** includes three components:
- **Memory Summarization:** Point-by-point summary of agent's current execution step with optimization suggestions
- **Memory Retrieval:** Retrieval of the most relevant historical steps
- **Long-term Memory:** Ongoing updated memory for recording long-term historical steps

**PRM-score Evaluation Prompt** evaluates candidate ActionStep nodes on:
- Progress Toward Goal
- Error and Stability
- TTS Efficiency
- Reflection Usage
- Loop Detection
- Contextual Awareness

Scoring: 9-10 (clearly advances goal) down to 0 (severe issues)

---

*Authors: Ningning Wang, Xavier Hu, Pai Liu, Yue Hou, Heyuan Huang, Shengyu Zhang, Jian Yang, Jiaheng Liu, Ge Zhang, Changwang Zhang, Jun Wang, Yuchen Eleanor Jiang. Corresponding: He Zhu, Wangchunshu Zhou.*
