# Vibe Analytics: Principle 2 - Specification Over Execution

[← Back to Vibe Analytics](Vibe-Analytics)

---

## The Specification Paradigm

In the Vibe Analytics model, the AI handles the heavy lifting—from querying databases to generating visualizations and explanatory narratives. The analyst's primary responsibility becomes the **specification of requirements** with sufficient clarity that a machine can execute them without human intervention.

This requires:

1. **Precision**: Describing edge cases, thresholds, and business rules explicitly
2. **Anticipation**: Predicting failure modes and encoding fallback logic
3. **Context**: Providing domain knowledge that prevents the AI from generating technically correct but business-nonsensical outputs

> **Ready-to-use templates:** For actionable HFIS templates covering A/B tests, dashboards, data pipelines, statistical analysis plans, and more, see the [HFIS Practical Guide](Vibe-Analytics-HFIS-Practical-Guide).
>
> **Advanced practitioners:** For the Claims-Evidence-Failure Penalties tripartite system, Tool-Shaped vs. Colleague-Shaped AI personas, and production governance frameworks, see the [HFIS Technical Deep Dive](Vibe-Analytics-High-Fidelity-Intent-Specifications).

---

## The Art of the Perfect Prompt

The most valuable skill for an analyst in this paradigm is the ability to describe what needs to exist clearly enough that a machine can build it **without asking a follow-up question**.

### Poor vs. Excellent Specifications

**❌ Poor Specification:**
> "Show me campaign performance for Q4"

**Problems:**
- No granularity specified (daily? weekly? monthly?)
- No segmentation defined (all customers? by channel? by product?)
- No success metrics identified (opens? clicks? conversions? revenue?)
- No baseline for comparison (vs. Q3? vs. Q4 last year? vs. forecast?)

**✅ Excellent Specification:**
> "Generate a weekly cohort analysis of email campaign performance for Q4 2025, segmented by investor AUM quintile. For each cohort, calculate 7-day, 14-day, and 30-day conversion rates to funded account status. Flag any week where conversion rates drop more than 15% below the rolling 8-week average. During market volatility events (defined as S&P 500 daily movement >2%), annotate the time series with a marker and exclude those weeks from baseline calculations. Export to a Google Sheet with conditional formatting highlighting statistically significant changes (p<0.05, two-tailed t-test)."

**What makes it excellent:**
- **Temporal granularity**: Weekly cohorts
- **Segmentation logic**: AUM quintiles
- **Conversion windows**: 7/14/30 days
- **Anomaly detection**: 15% threshold, 8-week baseline
- **Edge cases**: Market volatility handling
- **Statistical rigor**: Significance testing parameters
- **Output format**: Google Sheet with formatting

---

## The Specification Framework

### 1. Define the Business Question

**Template:** "We need to understand [WHAT] in order to decide [WHY]"

**Examples:**
- "We need to understand which customer segments respond best to retirement planning messaging in order to decide budget allocation for Q2"
- "We need to understand if our new onboarding flow is increasing activation rates in order to decide whether to roll it out to all users"

### 2. Specify Data Sources and Timeframes

**What to include:**
- Exact table/database names (if known)
- Date ranges with timezone specifications
- Join keys and relationship logic
- Handling of missing data

**Example:**
```
Data sources:
- Email platform: Braze API, campaigns table, UTC timezone
- CRM: Salesforce, Account and Opportunity objects
- Conversions: Snowflake.analytics.funded_accounts, EST timezone

Timeframe: October 1, 2025 - December 31, 2025
Join logic: Email recipient email → CRM Account.Email → Snowflake user_id
Missing data handling: Exclude records with null email or user_id
```

### 3. Define Segmentation and Aggregation

**Critical elements:**
- Segmentation variables and breakpoints
- Aggregation level (user, account, campaign, day, week)
- Minimum sample size requirements

**Example:**
```
Segmentation:
- AUM quintiles: <$10K, $10K-$50K, $50K-$250K, $250K-$1M, >$1M
- Account age: <6 months, 6-24 months, 24+ months
- Product type: Brokerage only, IRA only, Both

Aggregation: Weekly cohorts (Sunday-Saturday)
Minimum sample size: 100 conversions per segment, otherwise flag as "insufficient data"
```

### 4. Encode Business Rules and Edge Cases

**This is where domain expertise becomes irreplaceable:**

**For Marketing to Retail Investors:**
```
Edge cases to handle:
1. Market volatility: If S&P 500 daily movement exceeds ±2%, annotate chart and exclude from baseline calculations
2. Tax season: Jan 1 - April 15, expect 30-40% lower email engagement, note in summary
3. Year-end: December behavior is anomalous due to tax-loss harvesting, treat separately
4. Regulatory events: If SEC/FINRA announces new rules, pause attribution for affected channels
5. Account funding lag: Conversions can occur up to 45 days after email send, not just 30 days
```

### 5. Specify Output Format and Visualization

**Be explicit about:**
- Chart types and axes
- Color schemes and annotations
- Executive summary length and tone
- Statistical notation preferences

**Example:**
```
Visualization requirements:
- Primary chart: Time series line chart, conversion rate on Y-axis (0-10%), weeks on X-axis
- One line per AUM quintile, use Vanguard brand colors
- Annotate market volatility events with vertical dashed lines
- Include 95% confidence intervals as shaded regions
- Secondary chart: Waterfall chart showing week-over-week changes

Executive summary:
- 3-5 bullet points, C-suite appropriate language
- Lead with most actionable insight
- Include statistical significance where relevant
- Avoid jargon (say "investor" not "user")
```

### 6. Define Success Criteria and Validation

**How will you know the output is correct?**

**Example:**
```
Validation checks:
- Total email sends should match Braze dashboard (±5% acceptable)
- Conversion rates should be between 0.5% and 5% (flag if outside range)
- Sum of quintile sample sizes should equal total sample (exact match)
- Statistical significance should be calculated using two-tailed t-test, not one-tailed
- Confidence intervals should never be negative

Acceptance criteria:
- All validation checks pass
- No missing data warnings for segments with >100 records
- Charts render correctly in Google Sheets
- Executive summary is <200 words
```

---

## Marketing-Specific Edge Cases

### For Self-Directed Investor Analytics

#### Market Conditions
```
Specification example:
"During bull markets (S&P 500 YTD return >15%), expect higher engagement with growth equity content.
During bear markets (S&P 500 YTD return <-10%), expect higher engagement with bond/stability content.
Adjust content recommendations accordingly."
```

#### Regulatory Constraints
```
Specification example:
"FINRA Rule 2210 prohibits projecting future performance. All email content must be reviewed for compliance.
Flag any AI-generated subject lines containing words: 'guaranteed', 'projected', 'expected returns', 'outperform'.
Require manual review before sending."
```

#### Lifecycle Stage
```
Specification example:
"Segment by lifecycle:
- Prospect: No funded account
- New: <$1K AUM or <90 days since first deposit
- Growing: $1K-$100K AUM, actively contributing
- Mature: >$100K AUM, low monthly activity
- At-risk: No login in 180 days, no contributions in 365 days

Apply different conversion funnels to each stage."
```

#### Asset Class Behavior
```
Specification example:
"Investors with >80% equity allocation are 3x more likely to engage during market volatility.
Investors with >60% bond allocation are 2x more likely to engage during rate announcements.
Personalize email send times based on portfolio composition and market events."
```

#### Seasonality
```
Specification example:
"Tax season (Jan-April): Open rates drop 25%, but conversion intent is highest. Optimize for quality over volume.
Enrollment periods (Nov-Dec): Engagement drops but contribution amounts spike. Emphasize deadlines.
Year-end planning (Oct-Dec): Highest information consumption, optimal time for educational content."
```

---

## Common Specification Failures

### Failure Mode 1: Ambiguous Success Metrics

**Bad:**
> "Analyze if the campaign was successful"

**Why it fails:** "Successful" is undefined. Opens? Clicks? Conversions? Revenue? ROI?

**Fix:**
> "Calculate incremental revenue attributable to the campaign using a holdout control group, comparing 30-day post-campaign revenue between treatment and control, adjusting for pre-campaign spend levels using CUPED variance reduction."

### Failure Mode 2: Missing Edge Case Handling

**Bad:**
> "Calculate average order value by segment"

**Why it fails:** What if a segment has zero orders? Outliers? Returns/refunds?

**Fix:**
> "Calculate median order value by segment (more robust to outliers). For segments with <50 orders, flag as 'insufficient data'. Exclude orders >$50K (likely data errors). Net of returns/refunds."

### Failure Mode 3: Insufficient Context for AI

**Bad:**
> "Find underperforming campaigns"

**Why it fails:** AI doesn't know your business thresholds, baselines, or strategic priorities.

**Fix:**
> "Identify campaigns in bottom quintile of ROI (cost per funded account). Underperforming = ROI >2 standard deviations below 6-month rolling average. Prioritize campaigns with >$10K spend for investigation."

### Failure Mode 4: No Validation Requirements

**Bad:**
> "Generate monthly dashboard"

**Why it fails:** No way to catch data errors, API failures, or logic bugs.

**Fix:**
> "Generate monthly dashboard. Validation: (1) Row counts match previous month ±10%, (2) Revenue totals reconcile with finance system, (3) Conversion rates between 0.1% and 10%, (4) No null values in primary metrics. Alert if any validation fails."

---

## Building Your Specification Library

### Reusable Templates

Create a library of specification templates for common analyses:

**Template: Email Campaign Performance**
```markdown
# Email Campaign Performance Analysis

## Objective
Evaluate email campaign effectiveness for [CAMPAIGN_NAME] sent [DATE_RANGE]

## Data Sources
- Email platform: [PLATFORM]
- Conversion tracking: [DATABASE]
- Customer attributes: [CRM]

## Segments
- [LIST_SEGMENTS]

## Metrics
Primary: [PRIMARY_METRIC]
Secondary: [SECONDARY_METRICS]
Guardrails: [GUARDRAIL_METRICS]

## Edge Cases
- Market volatility: [RULE]
- Seasonality: [RULE]
- Data quality: [RULE]

## Validation
- [VALIDATION_CHECK_1]
- [VALIDATION_CHECK_2]

## Output
- Format: [FORMAT]
- Visualization: [CHART_TYPES]
- Summary: [LENGTH, TONE]
```

### Collaborative Refinement

**Process:**
1. Analyst writes initial specification
2. AI attempts execution
3. AI flags ambiguities or missing information
4. Analyst refines specification
5. Repeat until AI can execute without clarifying questions
6. Save refined specification to library
7. Share with team for reuse

---

## The ROI of Better Specifications

### Time Investment

**Upfront cost:**
- 2-4 hours to write a comprehensive specification for a complex analysis
- May feel slower than "just writing the code yourself"

**Long-term payoff:**
- Specification is reusable for all similar future analyses
- 95%+ time savings on subsequent executions
- Eliminates "tribal knowledge" bottlenecks
- New team members can execute analyses day one

### Quality Improvements

**Well-specified analyses:**
- Fewer errors (validation is built-in)
- More consistent outputs (not analyst-dependent)
- Better documentation (specification is the documentation)
- Easier auditing (logic is explicit, not buried in code)

---

## Practice Exercises

### Exercise 1: Spec-ify This Request

**Stakeholder request:**
> "Can you pull together some data on how our new customers are doing?"

**Your task:** Write a comprehensive specification that includes:
- Define "new customers" (timeframe, account type)
- Define "how they're doing" (specific metrics)
- Specify segmentation, edge cases, validation, output format

### Exercise 2: Identify the Missing Pieces

**Incomplete specification:**
> "Calculate email campaign ROI by segment, flag underperformers, recommend optimizations"

**Your task:** List 10+ missing elements that would prevent an AI from executing this perfectly

### Exercise 3: Build Your First Template

**Your task:** Create a reusable specification template for your most common analytical task

---

## Next Steps

1. **Audit your most recent analysis** - Could it have been specified clearly enough for an AI to execute?
2. **Practice writing specifications** - Start with simple requests, build to complex
3. **Build your template library** - Document patterns for reuse
4. **Test with AI tools** - See where specifications break down and refine

---

[Continue to Principle 3: The Domain Translator →](Vibe-Analytics-Principle-3-Domain-Translator)

[← Back to Vibe Analytics](Vibe-Analytics)
