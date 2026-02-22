# Vibe Analytics: High-Fidelity Intent Specifications - Practical Guide for Marketing Analytics Teams

[← Back to Vibe Analytics](Vibe-Analytics) | [← HFIS Technical Deep Dive](Vibe-Analytics-High-Fidelity-Intent-Specifications)

---

## The Specification Bottleneck

**The single most important skill in the AI era is not prompting — it is the ability to define precisely what "good" looks like before any work begins.**

For decades, the bottleneck in analytics was execution speed — how fast a team could pull data, build models, or produce dashboards. AI has obliterated that constraint. The new bottleneck is **how clearly a team can articulate what it needs**.

> "Every vague instruction gets amplified by AI. If a spec or brief leaves room for interpretation, the model will fill it with confident nonsense." — Nate B. Jones

Consider how most A/B tests get scoped today: a stakeholder sends a Slack message saying "Can we test a new hero banner on the landing page?" An analyst translates this into a loose experiment brief, makes assumptions about success metrics, picks a sample size, and launches. Three weeks later, the results are ambiguous because no one pre-defined what "success" meant, which segments mattered, or what decision would follow each possible outcome. The specification was low-fidelity. The intent was never captured.

The antidote is **intent-driven specification** — the practice of ensuring every document serves a testable goal, enables a specific decision, and captures the "why" behind the "what." You're not just creating words — you're creating specifications for decisions. Each paragraph is part of the logic of the business, not a slot in a template.

A High-Fidelity Intent Specification (HFIS) captures not just requirements but **intent, success criteria, failure modes, constraints, and decision logic** at a level of precision sufficient for both humans and AI agents to execute against with minimal interpretation drift.

---

## Five Core Principles of the HFIS Methodology

These principles form the philosophical backbone of every high-fidelity specification.

### Principle 1: Define Intent for Every Document

Every specification must answer: **What decision will this enable? What action will someone take when this work is done?**

> "If a reader can't tell what the document is supposed to accomplish, it's useless, no matter how polished it sounds."

For an analytics team, this means an A/B test spec should not just describe the test — it should declare the business decision it will inform and the stakeholder who will make that decision.

**Test yourself:** Can you complete this sentence for your current project?

> "When this work is complete, [STAKEHOLDER NAME] will use the results to decide [SPECIFIC DECISION], and will take [SPECIFIC ACTION] based on the outcome."

If you can't fill in the blanks, your specification is incomplete.

---

### Principle 2: Specify Quality Criteria That Can Be Tested

Success cannot be a vibe. Specifications require **concrete, verifiable criteria** — not "improve engagement" but "increase click-through rate on the primary CTA by ≥8% at 95% statistical confidence within the existing customer segment."

Every output should have standards that can be measured, not debated. Every decision must have a name. Every action item an owner. Every open question a next step.

**Bad criteria:**
- "The dashboard should be useful"
- "The model should be accurate"
- "The test should show if the campaign works"

**Good criteria:**
- "A marketing director can identify the worst-performing campaign and its root cause within 90 seconds of opening the dashboard"
- "The model must achieve ≥0.78 AUC on a held-out test set AND the retention team must confirm that top-decile predictions are actionable within their existing contact strategy"
- "The test must detect a ≥8% relative lift in application completion rate at 95% confidence and 80% power within 28 days"

---

### Principle 3: Include Examples of Failure Alongside Success

**Maintain a "failure file"** — a collection of bad outputs that illustrate what to avoid.

> "Clarity about what bad looks like is often more useful than vague ideals of 'good writing.'"

For analytics, this means:
- A dashboard spec should include examples of misleading visualizations
- A test spec should describe what an inconclusive result looks like
- A pipeline spec should enumerate known failure modes
- A model spec should show examples of predictions that can't be operationalized

**The failure file is not optional.** It is the fastest path to shared understanding of quality standards.

---

### Principle 4: Use AI for Evaluation, Not Just Generation

Most teams use AI to produce drafts but never to check them. Use **AI-driven first-pass evaluations** against explicit quality criteria before human review.

> "The point is NOT to stop human reviews. The point is to make human reviews matter more."

**Implementation:**
- Feed draft specifications to AI and ask: "What assumptions am I making that aren't stated? What failure modes am I missing? What would a skeptical stakeholder challenge?"
- Use AI to check analytical outputs against the specification's success criteria before presenting to stakeholders
- Build automated evaluation prompts for recurring work products

**The philosophy:** 99.9% of attention is about to be LLM attention, not human attention. Build systems that focus scarce human attention on the right 0.1% of work.

---

### Principle 5: Make Every Output Help Someone Decide Something

This is the ultimate test of specification quality. **If a completed analysis, dashboard, or test result doesn't clearly enable a specific business decision, the specification failed.**

**Decision-oriented output design** works backward:
1. Start with the **decision** that needs to be made
2. Determine what **evidence** is needed to make that decision
3. Determine what **analysis** produces that evidence
4. Determine what **data and methodology** the analysis requires
5. Write the specification

Most teams work forward (data → analysis → output → hope someone decides something). HFIS works backward from the decision.

---

## Anatomy of a High-Fidelity Intent Specification

An HFIS for marketing analytics contains eight structural components. What distinguishes this from a traditional PRD or analytics brief is the emphasis on **intent preservation, testable criteria, explicit failure modes, and machine-readability**.

### Component 1: Intent Declaration

The specification opens with a concise statement of business intent — not what will be built, but **what decision will be enabled and why it matters now**.

**Required elements:**
- Business context and strategic question
- The stakeholder who owns the decision
- The deadline by which the decision must be made
- What happens if this work is not done (stakes)

**Purpose:** Externalizing tacit knowledge — writing down the unspoken expectations that normally live in people's heads.

**Example:**
```markdown
## Intent Declaration

BUSINESS QUESTION: Should we permanently change the credit card application
flow from 5 steps to 3 steps?

DECISION: Go/no-go on permanent UX change affecting 2.4M monthly applicants
DECISION-MAKER: VP of Digital Acquisition
DEADLINE: Q3 planning cycle (June 30, 2026)
STAKES: A successful change could increase annual new accounts by ~288K.
A failed change could increase abandonment and damage brand trust.
```

---

### Component 2: Success Criteria

Concrete, measurable definitions of success that can be evaluated objectively. **Binary pass/fail conditions, not aspirational goals.**

**For an A/B test:**
- Primary metric and minimum detectable effect size
- Required statistical power and confidence level
- Practical significance threshold (distinct from statistical significance)
- Minimum test duration

**For a dashboard:**
- Specific questions the dashboard must answer
- Behavioral criteria ("user can find X within Y seconds")
- Refresh frequency and maximum acceptable data latency

**For a model:**
- Model performance thresholds (AUC, precision, recall)
- Business operationalizability criteria
- Dual criterion: statistical quality AND business actionability

**Example:**
```markdown
## Success Criteria

PRIMARY METRIC: Application completion rate
MINIMUM DETECTABLE EFFECT: ≥8% relative lift
CONFIDENCE LEVEL: 95% (two-sided)
STATISTICAL POWER: 80%
MINIMUM DURATION: 14 days (captures weekly cyclicality)

PRACTICAL SIGNIFICANCE THRESHOLD:
A statistically significant lift of less than 5% relative does not
justify the engineering cost of the permanent change.

SECONDARY METRICS (monitored, not gating):
- Time-to-complete (expect decrease)
- Downstream approval rate (must not decrease >2%)
- 90-day account activation rate (must not decrease >3%)
```

---

### Component 3: Scope and Boundaries (Three-Tier System)

Explicit statements of what is in scope and what is out of scope, using a **three-tier boundary system**:

| Tier | Symbol | Meaning | Example |
|------|--------|---------|---------|
| **Always Do** | ✅ | Mandatory actions the executor must take | Segment results by new-to-bank vs. existing customers and by device type |
| **Ask First** | ⚠️ | Requires approval before proceeding | Extending test beyond planned 28-day window |
| **Never Do** | 🚫 | Hard constraints that must never be violated | Never include customers in active regulatory remediation cohorts |

**In financial services, the "never" tier is especially critical** for compliance and data governance constraints.

**Example:**
```markdown
## Scope and Boundaries

✅ ALWAYS:
- Segment results by new-to-bank vs. existing customers
- Segment by device type (mobile, desktop, tablet)
- Include Sample Ratio Mismatch check before analyzing results
- Document all covariates considered for inclusion

⚠️ ASK FIRST:
- Extending test beyond planned 28-day window
- Adding segments not in original specification
- Changing primary metric or success threshold
- Including data from external sources

🚫 NEVER:
- Include customers in active regulatory remediation cohorts
- Expose PII in test logs or analysis outputs
- Peek at results before reaching required sample size
- Report exploratory subgroup analyses as confirmatory findings
- Share raw customer-level data outside the analytics team
```

---

### Component 4: Methodology and Approach

How the work will be accomplished, including analytical framework, statistical methods, data sources, tools, and **explicit assumptions**.

**Key principle:** Rather than leaving methodology to the executor's judgment, the specification makes the approach explicit so deviations can be identified and discussed.

**Required elements:**
- Analytical framework or statistical method
- Data sources with access details
- Tools and environments
- Assumptions with rationale
- What would change if each assumption proved wrong

---

### Component 5: Failure Modes and Anti-Patterns

**The failure file applied to a specific project.** An enumeration of what bad looks like.

**For experiments:**
- Sample ratio mismatch indicating randomization failure
- Novelty effects in the first 48 hours (exclude from analysis)
- Interaction effects with concurrent promotional campaigns
- Metric tradeoffs requiring executive judgment (e.g., completions up but approval rates down)

**For dashboards:**
- Metrics without context (number without comparison point)
- Misleading y-axis scales
- Vanity metrics that look good but don't inform decisions
- Cramming 40 metrics onto a single view instead of progressive disclosure

**For models:**
- Predictions that are statistically excellent but operationally useless
- Features that leak future information into training data
- Class imbalance handling that inflates apparent performance
- Confounding treatment effects with macroeconomic shifts

**For pipelines:**
- Late-arriving data from upstream systems
- Schema changes in source tables
- Duplicate records from retry logic
- Timezone inconsistencies across source systems
- PII data flowing into non-PII-classified storage (financial services critical)

---

### Component 6: Output Format and Deliverables

A precise description of what the finished work product looks like, including format, structure, audience, and presentation context.

**Required:** Format and success criteria specified so executor can self-evaluate.

**Example:**
```markdown
## Output Format

DELIVERABLE 1: Executive Decision Brief
- Format: 1-page PDF or Slides
- Audience: VP of Digital Acquisition + CMO
- Content: Verdict, key finding, financial impact, recommendation
- Tone: Decisive, action-oriented, no statistical jargon
- Length: <500 words

DELIVERABLE 2: Technical Analysis Report
- Format: Jupyter notebook (reproducible)
- Audience: Analytics peers, data science review committee
- Content: Full methodology, diagnostics, code, data lineage
- Must include: SRM check results, power analysis, sensitivity analysis

DELIVERABLE 3: Stakeholder Presentation
- Format: Google Slides, 8-12 slides
- Structure: Context → Methodology → Results → Recommendation → Next Steps
- Visual style: Follow Hitchcock's Rules (establishing shot → detail)
```

---

### Component 7: Evidence and Rationale

The executor must document their reasoning — why they made specific choices, what alternatives they considered, and what assumptions they relied on.

**Purpose:** Creates an audit trail invaluable in regulated environments.

**Example:**
```markdown
## Evidence Requirements

The analyst must document:
1. WHY this statistical method was chosen over alternatives
2. WHAT alternative approaches were considered and rejected
3. WHICH assumptions the analysis relies on
4. HOW results would change if key assumptions were violated

Format: A short "why this design" section in the technical report,
structured as:
- Decision made → Rationale → Alternative considered → Why rejected
```

---

### Component 8: Validation and Review Checkpoints

Specific points in the workflow where human review is required before proceeding.

**Jones's "contract-first" method:** AI systematically asks clarifying questions until reaching a predefined confidence threshold before executing. This translates directly into staged review gates:

**Example checkpoints:**
```markdown
## Review Gates

GATE 1 (Before execution): Specification review
- Reviewer: Analytics manager
- Criteria: All 8 HFIS components complete, no ambiguities
- Timeline: Within 2 business days of submission

GATE 2 (During execution): Methodology check
- Reviewer: Peer data scientist
- Criteria: SRM check passed, data quality validated, assumptions documented
- Timeline: Before any results are interpreted

GATE 3 (Before delivery): Output review
- Reviewer: Analytics manager + compliance (if customer-facing)
- Criteria: Output matches specification, success criteria met or explicitly unmet
- Timeline: 1 business day before stakeholder presentation
```

---

## HFIS Templates for Marketing Analytics Use Cases

### Template 1: A/B Test and Experimentation Specification

This is the highest-leverage application of the HFIS methodology. Poorly specified experiments waste weeks of traffic, produce ambiguous results, and erode stakeholder trust.

```markdown
# Experiment Specification: [EXPERIMENT NAME]

## 1. Intent Declaration
BUSINESS QUESTION: [What business question does this answer?]
HYPOTHESIS: By [INTERVENTION], we predict a ≥[X]% [increase/decrease] in
[PRIMARY METRIC] among [TARGET POPULATION], based on [SUPPORTING DATA].
DECISION: [What go/no-go decision will this inform?]
DECISION-MAKER: [Name, title]
DEADLINE: [When must decision be made?]

## 2. Success Criteria
PRIMARY METRIC: [Metric name]
MINIMUM DETECTABLE EFFECT: ≥[X]% relative lift
CONFIDENCE LEVEL: [95%]
STATISTICAL POWER: [80%]
MINIMUM DURATION: [X days to capture cyclicality]
PRACTICAL SIGNIFICANCE: A lift of less than [X]% does not justify
[implementation cost / operational complexity].

SECONDARY METRICS (monitored, not gating):
- [Metric 1]: Must not [decrease/increase] by more than [X]%
- [Metric 2]: Monitored for directional signal
- [Metric 3]: Guardrail metric (experiment halts if violated)

## 3. Scope and Boundaries
✅ ALWAYS:
- Segment by [key dimensions]
- Run SRM check before analyzing
- Exclude first [48 hours] for novelty effect washout

⚠️ ASK FIRST:
- Extending beyond [planned duration]
- Adding segments not in original spec

🚫 NEVER:
- Include [excluded populations]
- Peek at results before [minimum sample/duration]
- Report exploratory subgroup analyses as confirmatory

## 4. Methodology
DESIGN: [RCT / Quasi-experimental / etc.]
RANDOMIZATION: [Unit, method, allocation ratio]
ANALYSIS: [Statistical test, covariates, adjustments]
MULTIPLE COMPARISONS: [Correction method if >3 segments]
OUTLIER HANDLING: [Winsorization / trimming / none]

## 5. Failure Modes
- Sample ratio mismatch (randomization failure)
- Novelty effects in first [48 hours]
- Interaction with concurrent campaigns: [list known overlaps]
- Metric tradeoff: [primary up but secondary down] requires
  executive judgment, not analyst judgment

## 6. Output Format
- Executive Decision Brief: [format, length, audience]
- Technical Report: [format, reproducibility requirements]
- Stakeholder Presentation: [format, slide count]

## 7. Evidence Requirements
- Document all modeling choices with rationale
- Distinguish confirmatory vs. exploratory analyses
- Pre-register analysis plan before data collection begins

## 8. Review Gates
- GATE 1: Spec review before launch
- GATE 2: SRM + data quality check at [50%] enrollment
- GATE 3: Output review before stakeholder presentation
```

---

### Template 2: Dashboard and Report Specification

Dashboard specs suffer from a chronic disease: stakeholders ask for "a dashboard showing our marketing performance" and analysts build something that technically shows data but enables no decisions. The HFIS approach inverts this by starting from decisions.

```markdown
# Dashboard Specification: [DASHBOARD NAME]

## 1. Intent Declaration
This dashboard supports the following SPECIFIC DECISIONS:
1. [Decision 1]: [Who makes it, how often, what data they need]
2. [Decision 2]: [Who makes it, how often, what data they need]
3. [Decision 3]: [Who makes it, how often, what data they need]

PRIMARY USER: [Role, name if known]
FREQUENCY OF USE: [Daily / Weekly / Monthly]

## 2. Success Criteria
BEHAVIORAL: [Primary user] can [identify specific insight] within
[X seconds] of opening the dashboard.
DATA FRESHNESS: Refreshed by [time] [timezone] [frequency]
LOAD TIME: Renders in under [X seconds]
ACCESSIBILITY: [Compliance standards]

## 3. Scope and Boundaries
✅ ALWAYS:
- Show YoY and MoM comparisons for all KPIs
- Include data source timestamp on every view
- Provide drill-down from summary to detail

⚠️ ASK FIRST:
- Adding metrics not in original specification
- Changing visualization types or layout

🚫 NEVER:
- Display customer-level data without RBAC
- Use arbitrary color schemes (follow brand + Hitchcock's Rules)
- Show metrics without comparison context

## 4. Visual Hierarchy (Hitchcock's Rules)
ESTABLISHING SHOT: [Executive summary panel with primary KPIs]
PRIMARY FOCUS: [The single most important metric, largest on screen]
PROGRESSIVE DISCLOSURE: [Summary → Channel → Campaign → Creative]
COLOR LOGIC: Green = beating target, Red = missing, Gray = neutral

## 5. Failure Modes (Anti-Patterns to Avoid)
- Metrics without context (raw number, no benchmark)
- Misleading y-axis (not starting at zero for bar charts)
- Vanity metrics prominent, decision metrics buried
- 40+ metrics on one screen (cognitive overload)
- Beautiful design, zero actionability

## 6. Data Sources
[Source table → Metric → Transformation logic → Refresh schedule]

## 7. Review Gates
- GATE 1: Wireframe review with primary user
- GATE 2: Data validation (metrics match source of truth)
- GATE 3: Usability test (can user find key insight in <90 seconds?)
```

---

### Template 3: Analytics Project Scoping Document

Larger analytics projects — propensity models, attribution analyses, CLV studies — require HFIS documents that function as both a project charter and a technical specification. **Written as if briefing both a senior executive and a junior data scientist.**

```markdown
# Analytics Project Specification: [PROJECT NAME]

## 1. Intent Declaration
BUSINESS CAPABILITY: [What capability is being created?]
Example: "Predict which credit card holders are likely to attrite within
90 days so that the retention team can intervene with targeted offers."

VALUE AT STAKE: [Quantified business impact]
Example: "$14.2M in annual revenue from preventable attrition (FY25 actuals)"

ORGANIZATIONAL COMMITMENT:
- [X] weeks of [role]'s time
- Access to [specific data sources]
- Stakeholder review at [checkpoints]

## 2. Success Criteria (Dual Criterion)
MODEL PERFORMANCE: [AUC ≥ X on held-out test set]
BUSINESS OPERATIONALIZABILITY: [Business team confirms top-decile
predictions are actionable within existing contact strategy]

Both criteria must be met. A statistically excellent model that
produces predictions the business cannot operationalize is a failure.

## 3. Methodology (with Explicit Assumptions)
FRAMEWORK: [CRISP-DM / custom]

For each modeling choice, document:
- CHOICE: [What was decided]
- RATIONALE: [Why this approach]
- ASSUMPTION: [What must be true for this to work]
- SENSITIVITY: [What changes if assumption is wrong]

## 4. Failure Modes
- Model overfits to historical patterns that won't persist
- Features leak future information into training data
- Class imbalance handling inflates apparent performance
- Predictions are accurate but not actionable by business team
- Model perpetuates historical biases in targeting

## 5-8. [Output, Evidence, Review Gates per standard HFIS structure]
```

---

### Template 4: Data Pipeline Specification

Data pipeline specs in financial services carry particular weight because of regulatory requirements around data lineage, audit trails, and change management.

```markdown
# Pipeline Specification: [PIPELINE NAME]

## 1. Intent Declaration
BUSINESS DEPENDENCY: [What business process depends on this pipeline?]
FAILURE IMPACT: [What happens if it fails or produces bad data?]
Example: "This pipeline feeds the weekly media mix model. A failure or
data quality issue undetected for >24 hours could result in $200K+ of
misallocated media spend."

## 2. Success Criteria
DATA COMPLETENESS: ≥[99.5]% of expected records
LATENCY: End-to-end processing within [X hours] of source availability
QUALITY RULES: [Referential integrity, range validation, dedup]
RECOVERY TIME: Pipeline can be rerun from scratch within [X hours]

## 3. Failure Modes (Systematic Enumeration)
| Failure Mode | Detection Mechanism | Remediation |
|---|---|---|
| Late-arriving upstream data | Freshness check at T+[X]hrs | Alert + retry at T+[Y]hrs |
| Schema change in source table | Schema drift detection | Block + alert data engineering |
| Duplicate records from retry | Dedup on [key columns] | Auto-dedup + log count |
| Timezone inconsistency | UTC normalization check | Convert all to UTC at ingestion |
| PII in non-PII storage | Column-level classification scan | Quarantine + alert compliance |

## 4. Data Lineage (Regulatory Requirement)
SOURCE SYSTEM → EXTRACTION METHOD → TRANSFORMATION LOGIC → DESTINATION

Every business rule applied during transformation must be documented.
Regulators expect to trace any reported number back to source data
through a documented chain.

## 5-8. [Remaining HFIS components per standard structure]
```

---

### Template 5: Statistical Analysis Plan (SAP)

A Statistical Analysis Plan is the most inherently specification-like document in analytics. The HFIS discipline adds intent and failure-mode rigor to the standard SAP structure.

```markdown
# Statistical Analysis Plan: [STUDY NAME]

## 1. Intent Declaration (Dual Language)
BUSINESS QUESTION: "Is our new email creative driving incremental
revenue, or are we cannibalizing purchases that would have happened anyway?"

STATISTICAL QUESTION: "Is the average incremental revenue per customer
in the treatment group significantly greater than zero after controlling
for baseline purchase propensity?"

## 2. Success Criteria (Inferential Framework)
PRIMARY ESTIMAND: Average treatment effect on the treated (ATT)
STATISTICAL TEST: Two-sided t-test with Welch's correction
SIGNIFICANCE LEVEL: α = 0.05
MULTIPLE COMPARISONS: Benjamini-Hochberg (if >3 segments)
PRACTICAL SIGNIFICANCE: Incremental revenue must exceed $2.50 per
customer to justify campaign cost

## 3. Pre-Registration
The following are LOCKED before data collection:
- Covariates included in regression: [list]
- Outlier handling: Winsorization at 1st and 99th percentiles
- Subgroup analyses: [list — these are CONFIRMATORY]

The following are EXPLORATORY (hypothesis-generating only):
- [Additional subgroups not pre-specified]
- [Post-hoc interaction analyses]

CRITICAL: Mixing confirmatory and exploratory analyses is a failure mode.
Confirmatory analyses answer pre-specified questions.
Exploratory analyses generate hypotheses for future testing.

## 4. Failure Modes Specific to Marketing SAPs
- P-hacking through undisclosed multiple comparisons
- Survivorship bias from excluding churned customers
- Confounding with macroeconomic shifts (rate changes, market volatility)
- Simpson's Paradox across segments
- Reporting exploratory findings as if they were pre-registered

## 5-8. [Remaining HFIS components per standard structure]
```

---

### Template 6: Marketing Mix Modeling and Attribution Specification

Attribution and MMM projects are high-stakes, technically complex, and frequently under-specified. The common failure: a model is built, results are presented, and the CMO asks "But what should I actually do differently?" — revealing that the specification never connected model output to budget decisions.

```markdown
# MMM/Attribution Specification: [MODEL NAME]

## 1. Intent Declaration
DECISION: This model will inform the FY27 media budget allocation
across [X] channels.
OUTPUT REQUIREMENT: Channel-level marginal ROI curves with confidence
intervals sufficient for the CFO to approve reallocation of up to $[X]M.
DECISION-MAKER: CMO + CFO (joint approval)
DEADLINE: [Budget planning cycle date]

## 2. Success Criteria (Dual)
MODEL FIT: MAPE ≤[X]%, R² ≥[X], out-of-sample forecast accuracy ≥[X]%
DECISION SUPPORT: Channel-level spend recommendations with sufficiently
narrow confidence intervals that directional recommendation does not
change across the 80% credible interval.

Both criteria must be met. An accurate model that produces
recommendations too uncertain to act on is a failure.

## 3. Scope
CHANNELS INCLUDED: [List all X channels with data sources]
TIME PERIOD: [Start] to [End]
GRANULARITY: [Weekly / Daily]
GEOGRAPHIC SCOPE: [National / Regional]
CONTROL VARIABLES: [Seasonality, macroeconomic, competitive]

## 4. Failure Modes
- Model captures correlation, not causation (addressable with
  experimental calibration)
- Confidence intervals too wide for actionable recommendations
- Model reflects historical spend patterns, not marginal returns
- Confounding between channels (attribution double-counting)
- Results sensitive to prior specification (Bayesian MMM)

## 5-8. [Remaining HFIS components per standard structure]
```

---

## Implementing HFIS on Your Team

Adopting HFIS is not a tool rollout — it is a **cultural shift** in how a team thinks about work before starting it.

> "The real skill isn't prompt engineering. It's knowing what you want. When you can't get what you want from AI, the issue is usually that you can't define what you want clearly enough for yourself. Not for the AI — for you."

### Step 1: Start with the Clarifying Questions Ritual

Before any analyst begins work, they should use clarifying questions — either with an AI assistant or with the requesting stakeholder:

> "Before we start this project, I need to understand: What does the end result look like? What criteria define success? What constraints exist? What would make this fail?"

**Key rule:** Ask one question at a time and build each subsequent question on the previous answer. This ritual alone surfaces 80% of implicit assumptions that cause rework.

---

### Step 2: Build Your Team's Failure File

Collect examples of analytics work that missed the mark and document specifically why they failed:

| Failed Work Product | What Went Wrong | Root Cause | HFIS Component That Would Have Prevented It |
|---|---|---|---|
| Dashboard nobody used | Showed data but enabled no decisions | No intent declaration — never defined which decisions it supported | Intent Declaration |
| A/B test with ambiguous results | Success metric wasn't defined upfront | No success criteria — "improve engagement" wasn't testable | Success Criteria |
| Model that couldn't be operationalized | Predictions accurate but business team couldn't act on them | No dual criterion — only specified model fit, not business actionability | Success Criteria (Dual) |
| Pipeline that delivered wrong data for 3 days | Timezone mismatch between source systems | No failure mode enumeration | Failure Modes |
| Report that caused compliance issue | Analyst included restricted customer data | No "never" tier boundary | Scope & Boundaries |

> "Clarity about what bad looks like is often more useful than vague ideals of 'good writing.'"

---

### Step 3: Adopt the Three-Tier Boundary System

For every project, explicitly define the three tiers:
- ✅ **Always do** (mandatory guardrails)
- ⚠️ **Ask first** (requires approval to deviate)
- 🚫 **Never do** (hard constraints, compliance, governance)

**In financial services, the "never" tier is where compliance, data governance, and regulatory constraints live.** Making these explicit in every specification — not buried in a policy manual — prevents costly violations and creates auditable documentation.

---

### Step 4: Use AI to Evaluate Specifications Before Execution

Feed your draft HFIS to an AI assistant and ask:

**Evaluation prompt:**
```
Review this specification against HFIS standards:

1. Is the intent declaration specific enough to identify the decision,
   decision-maker, and deadline?
2. Are success criteria binary (pass/fail) and testable?
3. Are scope boundaries defined using the three-tier system
   (always/ask/never)?
4. Are failure modes enumerated with detection and remediation?
5. Is the output format specified precisely enough for the executor
   to self-evaluate?
6. What assumptions am I making that aren't stated?
7. What would a skeptical stakeholder challenge?
8. What failure modes am I missing?
```

**The specification improves before any analytical work begins.**

---

### Step 5: Version-Control Your Specifications

Every HFIS should live in a shared repository, version-controlled so that changes to scope, methodology, or success criteria are tracked.

**Benefits:**
- When results don't match expectations, spec history reveals whether intent drifted during execution
- Changes to success criteria after data collection begins are visible and auditable
- New team members can learn from specification evolution over time
- Regulatory reviews can trace any decision back to its pre-registered specification

> "Specs give teams something to version that reflects what they meant to build, not just the files that were generated."

---

## Why Financial Services Teams Need This More Than Most

### The Regulatory Amplifier

When an analytics output informs a decision that affects customers — credit offers, risk pricing, marketing targeting — that output may be subject to regulatory scrutiny. A regulator wants to see the chain:

```
Business question → Methodology → Data → Analysis → Decision
```

An HFIS provides exactly this chain, documented before work began and traceable after it concluded.

**Regulatory corollary:** Regulators have always wanted tacit knowledge externalized. Most firms have been failing at it for decades. The unspoken expectations about model validation, permissible data sources, and fair targeting have lived in senior analysts' heads. HFIS documents make these expectations explicit, reviewable, and auditable — satisfying both quality frameworks and regulatory expectations around:

- **Model risk management** (SR 11-7 / OCC 2011-12)
- **Fair lending** compliance
- **Data governance** requirements
- **Marketing compliance** (FINRA 2210, SEC Marketing Rule)

### The Interpretation Drift Problem

When multiple analysts work on related projects using vague briefs, they make different assumptions, use different methodologies, and produce inconsistent results. HFIS documents create a shared specification that eliminates this drift.

**In a financial services firm running dozens of concurrent experiments and analytics projects, consistency is not a luxury — it is a prerequisite for trustworthy decision-making.**

---

## The Fundamental Reorientation

The shift is not about better prompts or better tools. It is about a **fundamental reorientation of where intellectual effort is invested** — moving it from execution (which AI can increasingly handle) to specification (which remains irreducibly human).

> "Most people spend their entire careers never learning to clearly define what 'good' looks like in their work. They operate on vibes. AI just made this impossible to sustain."

For a marketing analytics team, adopting HFIS means **slower starts and faster finishes**. The time spent writing a rigorous specification before launching an experiment, building a dashboard, or scoping a model is repaid many times over in:

- **Reduced rework** (ambiguity caught upfront, not at delivery)
- **Clearer results** (success criteria defined before data collection)
- **Faster stakeholder alignment** (intent captured, not assumed)
- **Stronger regulatory defensibility** (full audit trail from intent to decision)

**The specification is not overhead. It is the work.** Once you can define "good," the technical execution becomes almost trivial.

---

## Further Reading

- [HFIS Technical Deep Dive](Vibe-Analytics-High-Fidelity-Intent-Specifications) — Claims-Evidence-Failure Penalties, Tool/Colleague-Shaped agents, governance frameworks
- [Principle 2: Specification Over Execution](Vibe-Analytics-Principle-2-Specification-Over-Execution) — Core specification concepts and frameworks
- [Principle 3: The Domain Translator](Vibe-Analytics-Principle-3-Domain-Translator) — The human expertise that makes specifications valuable
- [Risks and Mitigation](Vibe-Analytics-Risks-and-Mitigation) — Managing quality degradation and organizational resistance

---

[← Back to Vibe Analytics](Vibe-Analytics)
