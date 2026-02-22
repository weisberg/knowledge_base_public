# Anthropic Engineering Knowledge Base: The Definitive Reference for AI-Powered Analytics Teams

**This document synthesizes every article from Anthropic's engineering blog and adjacent research publications into an organized, thematic reference.** It covers 14+ articles spanning December 2024 through January 2026, plus Anthropic's official documentation on prompt engineering, evaluations, and deployment. Each section is structured for data analysts and teams building AI-focused analytics workflows: plain-English summaries first, then technical deep-dives, then verbatim examples and templates.

---

## Table of Contents

1. [Context Engineering & Prompt Design](#1-context-engineering--prompt-design)
2. [Agentic Systems & Architecture Patterns](#2-agentic-systems--architecture-patterns)
3. [Tool Use, MCP & Advanced Orchestration](#3-tool-use-mcp--advanced-orchestration)
4. [Agent Skills & the SKILL.md Pattern](#4-agent-skills--the-skillmd-pattern)
5. [Multi-Agent Systems & Research Architecture](#5-multi-agent-systems--research-architecture)
6. [The Claude Agent SDK & Claude Code](#6-the-claude-agent-sdk--claude-code)
7. [Evaluation Frameworks & Benchmarking](#7-evaluation-frameworks--benchmarking)
8. [Safety, Alignment & Interpretability Engineering](#8-safety-alignment--interpretability-engineering)
9. [Inference, Deployment & Cost Optimization](#9-inference-deployment--cost-optimization)
10. [How AI Is Transforming Engineering Work](#10-how-ai-is-transforming-engineering-work)
11. [Practical Templates, Prompt Examples & Code Snippets](#11-practical-templates-prompt-examples--code-snippets)
12. [Master Reference Table of All Articles](#12-master-reference-table-of-all-articles)

---

## 1. Context Engineering & Prompt Design

### Overview

Anthropic's single most important conceptual contribution to practical AI engineering is the shift from **prompt engineering** to **context engineering**. Where prompt engineering focuses on crafting the text of a single instruction, context engineering manages the entire universe of tokens—system prompts, tools, examples, message history, retrieved data, and MCP state—that flows into a model's limited attention window across many inference turns.

This section synthesizes two primary sources: the engineering blog article "Effective context engineering for AI agents" (September 2025, by Prithvi Rajasekaran, Ethan Dixon, Carly Ryan, and Jeremy Hadfield) and Anthropic's official prompt engineering documentation.

---

### "Effective context engineering for AI agents"

**Plain-English summary:** As AI agents operate over longer tasks with more tools, the old approach of writing one great prompt is no longer sufficient. Context engineering is about curating the smallest, highest-signal set of tokens that maximizes the chance of a good outcome—across every turn of a multi-turn agent loop. The article introduces the concept of "context rot" (model accuracy degrades as token count grows due to the transformer architecture's quadratic attention mechanism) and provides three techniques for handling tasks that exceed the context window: compaction, structured note-taking, and sub-agent architectures.

**Key finding:** The guiding principle is to find **the smallest possible set of high-signal tokens that maximize the likelihood of a desired outcome**. Every token depletes the model's finite "attention budget."

**Technical deep-dive:**

**Context rot** stems from the transformer architecture where every token attends to every other token (n² pairwise relationships). Models develop attention patterns from training data where shorter sequences are common. Position encoding interpolation allows handling longer sequences but with degradation—creating a performance gradient rather than a hard cliff.

**The Goldilocks zone for system prompts** sits between two failure modes. Too prescriptive: hardcoded if-else logic that creates fragility. Too vague: high-level guidance without concrete signals. The optimal zone provides strong heuristics—specific enough to guide behavior, flexible enough to generalize.

**Three long-horizon techniques:**

**Compaction** summarizes the conversation when nearing context limits, then reinitiates with the summary. In Claude Code, the model preserves architectural decisions, unresolved bugs, and implementation details while discarding redundant tool outputs. The agent continues with compressed context plus the five most recently accessed files. Best practice: start by maximizing recall in summaries, then iterate to improve precision.

**Structured note-taking** has the agent regularly persist notes to external memory. Anthropic demonstrated this with Claude playing Pokémon, where the agent maintained precise tallies across thousands of game steps, tracking objectives like "for the last 1,234 steps I've been training my Pokémon in Route 1, Pikachu has gained 8 levels toward the target of 10." A file-based memory tool was released in public beta on the Claude Developer Platform as part of the Sonnet 4.5 launch.

**Sub-agent architectures** assign specialized sub-agents to focused tasks with clean context windows. Each sub-agent may use tens of thousands of tokens internally but returns only a **1,000–2,000 token condensed summary** to the orchestrator.

**Just-in-time context retrieval** is the recommended default over pre-loading. Agents maintain lightweight identifiers (file paths, stored queries, web links) and dynamically load data at runtime. Claude Code uses CLAUDE.md files loaded upfront combined with `glob` and `grep` for just-in-time navigation—a **hybrid strategy** that works especially well for less dynamic content like legal or finance work.

**Practical recommendations for analytics teams:**
- Organize prompts into distinct sections using XML tags (`<background_information>`, `<instructions>`) or Markdown headers (`## Tool guidance`)
- Curate diverse, canonical few-shot examples rather than a laundry list of edge cases—**"examples are the 'pictures' worth a thousand words"**
- Tools should be self-contained, robust to error, and extremely clear on intended use
- If a human engineer can't definitively say which tool should be used, an AI agent can't do better
- Start testing a minimal prompt with the best model, then add instructions based on failure modes
- "Do the simplest thing that works" remains the best advice

---

### Anthropic's official prompt engineering techniques

Anthropic's documentation at `docs.anthropic.com` provides 13 technique pages. The key techniques relevant to analytics teams:

**Be clear and direct.** The golden rule: show your prompt to a colleague with minimal context—if they're confused, Claude will be too. Think of Claude as "a brilliant but very new employee (with amnesia) who needs explicit instructions." Provide: what results will be used for, target audience, workflow position, and end goal.

**Multishot prompting (few-shot examples).** Include **3–5 diverse, relevant examples**. Examples should be wrapped in `<example>` tags nested within `<examples>`. Benefits: reduces misinterpretation, enforces uniform output structure, boosts complex task handling. You can ask Claude to evaluate examples for relevance, diversity, and clarity, or to generate additional ones.

**Chain of thought (CoT).** The simplest approach: include "Think step by step" in the prompt. Structure thinking in XML with `<thinking>` and `<answer>` tags. When extended thinking is enabled, use high-level instructions rather than prescriptive step-by-step guidance.

**XML tags.** Use tags like `<instructions>`, `<example>`, `<formatting>` to separate prompt parts. Be consistent with tag names. Nest tags for hierarchy. Combine with multishot (`<examples>`) and CoT (`<thinking>`, `<answer>`). This prevents Claude from mixing up instructions with examples or context.

**Long context tips.** Place documents (~20K+ tokens) at the top, above queries/instructions/examples—queries at the end improve quality by **up to 30%**. Structure documents with indexed XML:

```xml
<documents>
  <document index="1">
    <source>annual_report_2023.pdf</source>
    <document_content>
      {{ANNUAL_REPORT}}
    </document_content>
  </document>
</documents>
```

**Extended thinking tips.** Minimum budget: 1,024 tokens. Start small and increase incrementally. Extended thinking performs best in English; final outputs can be in any language. For verification, ask Claude to verify with test cases before declaring complete. Do NOT pass Claude's extended thinking back in user text blocks—this degrades results.

**Claude 4 best practices.** When extended thinking is disabled, Claude Opus 4.5 is particularly sensitive to the word "think"—replace with "consider," "believe," or "evaluate." To constrain excessive deliberation: "Prioritize execution over deliberation. Choose one approach and start producing output immediately."

---

## 2. Agentic Systems & Architecture Patterns

### "Building effective agents" — December 19, 2024

*By Erik Schluntz and Barry Zhang. This is Anthropic's foundational article on agent design and the most widely referenced piece in the engineering blog.*

**Plain-English summary:** After working with dozens of teams building AI agents across industries, Anthropic found that the most successful implementations use simple, composable patterns—not complex frameworks. The article defines the difference between "workflows" (predefined code paths) and "agents" (LLMs dynamically directing their own processes), then presents five workflow patterns and autonomous agents, with clear guidance on when each is appropriate.

**Core architectural distinction:**

| Type | Definition | Best for |
|------|-----------|----------|
| **Workflow** | LLMs + tools orchestrated through predefined code paths | Well-defined tasks needing predictability |
| **Agent** | LLMs dynamically directing their own processes and tool usage | Open-ended problems needing flexibility |

**The building block: the Augmented LLM** — An LLM enhanced with retrieval, tools, and memory. All agentic systems start here.

**Five workflow patterns:**

**1. Prompt chaining** decomposes a task into a sequence of steps where each LLM call processes the output of the previous one, with programmatic "gates" for verification between steps. Use for tasks that cleanly decompose into fixed subtasks. *Analytics example: Generate SQL query → validate it → execute → summarize results.*

**2. Routing** classifies an input and directs it to a specialized followup task. Separation of concerns means each path has an optimized prompt. Use when distinct categories benefit from separate handling. *Analytics example: Route easy metric lookups to Haiku (fast/cheap), complex analysis to Sonnet (capable), and ambiguous queries to human review.*

**3. Parallelization** has LLMs work simultaneously with outputs aggregated. Two variations: **sectioning** (independent subtasks in parallel) and **voting** (same task multiple times for confidence). *Analytics example: Run a guardrail check on user input while simultaneously generating the analysis.*

**4. Orchestrator-workers** has a central LLM dynamically break down tasks, delegate to worker LLMs, and synthesize results. Unlike parallelization, subtasks aren't pre-defined. *Analytics example: A research query that needs data from Salesforce, internal databases, and web sources simultaneously.*

**5. Evaluator-optimizer** has one LLM generate a response while another provides evaluation and feedback in a loop until quality thresholds are met. *Analytics example: Generate a data visualization → evaluate it for accuracy and clarity → refine iteratively.*

**Autonomous agents** use tools based on environmental feedback in a loop. They plan, operate independently, and gain "ground truth" from the environment at each step (tool call results, code execution output). **Key warning: higher costs and potential for compounding errors.** Anthropic recommends extensive testing in sandboxed environments.

**Three core principles:** (1) Maintain **simplicity** in design, (2) Prioritize **transparency** by showing planning steps, (3) Carefully craft the **Agent-Computer Interface (ACI)** through thorough tool documentation and testing.

**Critical ACI insight from SWE-bench:** "We actually spent more time optimizing our tools than the overall prompt." Example: the model made mistakes with relative filepaths; switching to absolute filepaths eliminated the errors. Treat tool descriptions like docstrings for a junior developer.

---

## 3. Tool Use, MCP & Advanced Orchestration

### "Writing effective tools for AI agents—using AI agents" — October 2025

*By Ken Aizawa with contributions across Research, MCP, Product Engineering, Marketing, Design, and Applied AI teams.*

**Plain-English summary:** Tools are a "new kind of software"—a contract between deterministic systems and non-deterministic agents. This article describes Anthropic's evaluation-driven methodology for improving tool performance: prototype → evaluate → analyze → refine → repeat. The key insight is that even small refinements to tool descriptions yield dramatic improvements. Claude Sonnet 3.5 achieved **state-of-the-art SWE-bench Verified performance** after precise tool description refinements alone.

**Five principles for writing effective tools:**

**1. Choose the right tools.** Build few, thoughtful tools targeting high-impact workflows. Consolidate related operations. Instead of `list_users` + `list_events` + `create_event`, implement `schedule_event` that finds availability and schedules. Instead of `read_logs`, implement `search_logs` that returns only relevant lines with context. Instead of `get_customer_by_id` + `list_transactions` + `list_notes`, implement `get_customer_context` that compiles all relevant info at once.

**2. Namespace your tools.** Group related tools under common prefixes (e.g., `asana_search`, `jira_search`, `asana_projects_search`). Prefix vs. suffix namespacing has **non-trivial effects** on tool-use evaluations—test both approaches.

**3. Return meaningful context.** Prioritize contextual relevance over flexibility. Use human-readable fields (`name`, `image_url`, `file_type`) not cryptic identifiers (`uuid`, `256px_image_url`, `mime_type`). Resolve UUIDs to semantically meaningful language.

**4. Optimize for token efficiency.** Implement pagination, range selection, filtering, and truncation with sensible defaults. Claude Code restricts tool responses to **25,000 tokens by default**. Expose a `response_format` enum (concise/detailed) to let agents control verbosity—"detailed" responses use ~206 tokens vs. ~72 for "concise" (roughly 3×).

**5. Prompt-engineer tool descriptions.** Describe tools as you would to a new hire. Make implicit knowledge explicit. Avoid ambiguity. Name parameters unambiguously (`user_id` not `user`).

**Using agents to improve tools:** Concatenate evaluation transcripts and paste into Claude Code. Claude excels at analyzing transcripts and refactoring tools simultaneously. A **tool-testing agent** that repeatedly tries using a flawed tool and then rewrites its description produced a **40% decrease in task completion time** for future agents.

**Evaluation design:** Generate tasks inspired by real-world use. Pair each with a verifiable outcome. Use simple agentic loops (while-loops wrapping alternating LLM API and tool calls). Instruct evaluation agents to output reasoning/feedback BEFORE tool calls to trigger CoT. Collect metrics: accuracy, total runtime, tool calls, token consumption, tool errors.

---

### "Advanced tool use" — November 2025

*By Bin Wu et al.*

**Plain-English summary:** Anthropic released three beta features that fundamentally change how agents interact with tools: Tool Search Tool for dynamic discovery, Programmatic Tool Calling for code-based orchestration, and Tool Use Examples for learning from concrete patterns. Together, they solve the problem of managing hundreds of tools without overwhelming the context window.

**Key metrics:**
- **Tool Search Tool:** 85% token reduction (77K → 8.7K tokens); accuracy improved from 49% → 74% (Opus 4) and 79.5% → 88.1% (Opus 4.5)
- **Programmatic Tool Calling:** 37% token reduction on complex research tasks; reduced 200KB raw data to 1KB results
- **Tool Use Examples:** 72% → 90% accuracy on complex parameter handling

**Tool Search Tool** enables on-demand tool discovery. Mark tools with `defer_loading: true` to exclude them from the initial prompt. Claude searches for tools when needed, loading only relevant ones. System prompt provides high-level guidance about available tool categories.

```json
{
  "tools": [
    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
    {
      "name": "github.createPullRequest",
      "description": "Create a pull request",
      "input_schema": {},
      "defer_loading": true
    }
  ]
}
```

**Prompt caching note:** Tool Search Tool doesn't break prompt caching because deferred tools are excluded from the initial prompt entirely.

**Programmatic Tool Calling (PTC)** lets Claude write Python code that orchestrates tools, with only final outputs entering context. Each traditional tool call requires a full inference pass; PTC replaces multiple passes with a single code execution.

```python
# Claude generates this orchestration code
team = await get_team_members("engineering")
levels = list(set(m["level"] for m in team))
budget_results = await asyncio.gather(*[
    get_budget_by_level(level) for level in levels
])
budgets = {level: budget for level, budget in zip(levels, budget_results)}
expenses = await asyncio.gather(*[
    get_expenses(m["id"], "Q3") for m in team
])
exceeded = []
for member, exp in zip(team, expenses):
    budget = budgets[member["level"]]
    total = sum(e["amount"] for e in exp)
    if total > budget["travel_limit"]:
        exceeded.append({"name": member["name"], "spent": total, "limit": budget["travel_limit"]})
print(json.dumps(exceeded))
```

**Tool Use Examples** embed concrete sample invocations in tool definitions via the `input_examples` field. Three patterns demonstrate the spectrum from full (critical bug with escalation) to partial (feature request) to minimal (internal task):

```json
{
  "name": "create_ticket",
  "input_schema": { "..." },
  "input_examples": [
    {
      "title": "Login page returns 500 error",
      "priority": "critical",
      "labels": ["bug", "authentication", "production"],
      "reporter": {"id": "USR-12345", "name": "Jane Smith",
        "contact": {"email": "jane@acme.com", "phone": "+1-555-0123"}},
      "due_date": "2024-11-06",
      "escalation": {"level": 2, "notify_manager": true, "sla_hours": 4}
    },
    {"title": "Add dark mode support", "labels": ["feature-request", "ui"],
      "reporter": {"id": "USR-67890", "name": "Alex Chen"}},
    {"title": "Update API documentation"}
  ]
}
```

**Combining all three features:**

```python
client.beta.messages.create(
    betas=["advanced-tool-use-2025-11-20"],
    model="claude-sonnet-4-5-20250929",
    max_tokens=4096,
    tools=[
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        {"type": "code_execution_20250825", "name": "code_execution"},
        # Your tools with defer_loading, allowed_callers, and input_examples
    ]
)
```

---

### "Code execution with MCP: Building more efficient AI agents" — November 2025

*By Adam Jones and Conor Kelly.*

**Plain-English summary:** As MCP adoption scales with thousands of connected servers, tool definitions and intermediate results overwhelm the context window. The solution: present MCP servers as code APIs on a filesystem rather than direct tool calls, letting agents write code to interact with tools. This reduces token usage from 150,000 to 2,000—a **98.7% reduction**. Cloudflare independently confirmed these findings, calling the approach "Code Mode."

**The filesystem approach:**

```
servers
├── google-drive
│   ├── getDocument.ts
│   └── index.ts
├── salesforce
│   ├── updateRecord.ts
│   └── index.ts
└── ...
```

Agents discover tools by exploring the filesystem (listing `./servers/`), reading only the definitions they need. The critical comparison shows that a Google Drive → Salesforce workflow that previously duplicated a full meeting transcript through the model context twice can instead be done in code where the transcript never enters the model:

```typescript
import * as gdrive from './servers/google-drive';
import * as salesforce from './servers/salesforce';

const transcript = (await gdrive.getDocument({ documentId: 'abc123' })).content;
await salesforce.updateRecord({
  objectType: 'SalesMeeting',
  recordId: '00Q5f000001abcXYZ',
  data: { Notes: transcript }
});
```

**Privacy-preserving operations:** Intermediate results stay in the execution environment. MCP clients can intercept and tokenize PII (emails → `[EMAIL_1]`, phones → `[PHONE_1]`). Real data flows between tools but never through the model.

**State persistence and reusable skills:** Agents can save working code as reusable functions in `./skills/` directory, adding SKILL.md files to create structured skill references.

---

## 4. Agent Skills & the SKILL.md Pattern

### "Equipping agents for the real world with Agent Skills" — October 2025

*By Barry Zhang, Keith Lazuka, and Mahesh Murag. Updated December 18, 2025: published as an open standard at agentskills.io.*

**Plain-English summary:** Agent Skills are organized folders of instructions, scripts, and resources that agents discover and load dynamically. Instead of building fragmented custom agents for each use case, anyone can specialize a general-purpose agent by packaging domain expertise into composable skills—like putting together an onboarding guide for a new hire. The core design principle is **progressive disclosure**: skills expose information in layers so the context window stays lean.

**Progressive disclosure (three levels):**

| Level | What loads | When |
|-------|-----------|------|
| **1** | Name + description metadata | At agent startup (always in system prompt) |
| **2** | Full SKILL.md body | On-demand when Claude determines relevance |
| **3+** | Additional referenced files | Only when needed for the specific subtask |

This makes the amount of context bundled into a skill **effectively unbounded**.

**SKILL.md structure:** Must begin with YAML frontmatter containing `name` and `description`. The body contains instructions, references to additional files, and optionally executable scripts.

**Context window sequence:** (1) System prompt + skill metadata + user message → (2) Claude triggers skill by reading SKILL.md via Bash tool → (3) Claude reads referenced files as needed → (4) Claude proceeds with task using loaded skill instructions.

Skills can include **executable code** for deterministic operations. Pre-written Python or Bash scripts run without loading the script or data into context. This is especially valuable for operations like sorting, PDF parsing, or data transformation where code is more reliable than token generation.

**Guidelines for building skills:**
- **Start with evaluation:** Run agents on representative tasks, observe gaps, build skills incrementally
- **Structure for scale:** Split unwieldy SKILL.md files into separate referenced files; keep mutually exclusive contexts separate
- **Think from Claude's perspective:** Monitor real usage; watch for unexpected trajectories; pay special attention to the name and description
- **Iterate with Claude:** Ask Claude to capture successful approaches and common mistakes into reusable context

**Security:** Install skills only from trusted sources. Audit code dependencies and bundled resources. Watch for instructions connecting to untrusted external network sources.

**Resources:**
- Open standard: https://agentskills.io/
- Skills docs: https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview
- Skills cookbook: https://github.com/anthropics/claude-cookbooks/tree/main/skills
- PDF skill example: https://github.com/anthropics/skills/tree/main/document-skills/pdf

---

## 5. Multi-Agent Systems & Research Architecture

### "How we built our multi-agent research system" — June 2025

*By Jeremy Hadfield, Barry Zhang, Kenneth Lien, Florian Scholz, Jeremy Fox, and Daniel Ford.*

**Plain-English summary:** Anthropic built a production multi-agent system for Claude's Research feature that searches across the web, Google Workspace, and integrations. A lead agent coordinates the process, spawning specialized sub-agents that work in parallel. The system outperformed a single-agent approach by **90.2%** on research tasks. Token usage alone explains **80% of performance variance**, with multi-agent systems consuming roughly **15× more tokens** than chat interactions. The article details 8 prompt engineering principles, evaluation approaches, and production reliability challenges.

**Architecture: Orchestrator-worker pattern**

1. User submits query → **LeadResearcher agent** (Claude Opus 4) analyzes, develops strategy
2. LeadResearcher saves plan to **Memory** (persistence outside context window)
3. LeadResearcher spawns **Subagents** (Claude Sonnet 4) with specific research tasks
4. Each subagent independently: performs searches → evaluates results using interleaved thinking → returns compressed findings
5. LeadResearcher synthesizes results, spawns additional subagents if needed
6. All findings passed to **CitationAgent** for source verification
7. Final research with citations returned to user

**Key metrics:**
- **90.2% improvement** over single-agent (Opus 4 lead + Sonnet 4 subagents vs. single Opus 4)
- **80% of BrowseComp performance variance** explained by token usage alone
- **95%** explained by three factors: token usage, tool calls, model choice
- **90% research time reduction** for complex queries via parallel tool calling
- Upgrading from Sonnet 3.7 to Sonnet 4 produced larger gains than doubling the token budget on 3.7

**The 8 prompt engineering principles:**

**1. Think like your agents.** Build simulations using the Console with exact prompts and tools. Watch agents step-by-step to reveal failure modes: continuing past sufficient results, using overly verbose queries, selecting wrong tools.

**2. Teach the orchestrator how to delegate.** Each subagent needs: (a) an objective, (b) an output format, (c) guidance on tools and sources, (d) clear task boundaries. Without detail, agents duplicate work or leave gaps.

**3. Scale effort to query complexity.** Embedded scaling rules:
- Simple fact-finding: 1 agent, 3–10 tool calls
- Direct comparisons: 2–4 subagents, 10–15 calls each
- Complex research: 10+ subagents with clearly divided responsibilities

**4. Tool design and selection are critical.** Explicit heuristics: examine all available tools first, match to user intent, use web for broad searches, prefer specialized tools when available.

**5. Let agents improve themselves.** A tool-testing agent that attempts to use a flawed MCP tool dozens of times then rewrites the description produced a **40% decrease in task completion time** for future agents.

**6. Start wide, then narrow down.** Agents default to overly specific, long queries. Counter by prompting to start with short broad queries, evaluate, then progressively narrow.

**7. Guide the thinking process.** Extended thinking serves as controllable scratchpad. Lead agent uses it to plan approach, assess tools, determine query complexity and subagent count. Subagents use interleaved thinking after tool results to evaluate quality and identify gaps.

**8. Parallel tool calling transforms speed.** Two kinds: lead agent spins up 3–5 subagents in parallel; each subagent uses 3+ tools in parallel. This cut research time by **up to 90%** for complex queries.

**Example parallel tool use prompt (from Anthropic's open-source cookbook):**

```xml
<use_parallel_tool_calls>
For maximum efficiency, whenever you need to perform multiple independent
operations, invoke all relevant tools simultaneously rather than sequentially.
Call tools in parallel to run subagents at the same time. You MUST use parallel
tool calls for creating multiple subagents (typically running 3 subagents at
the same time) at the start of the research, unless it is a straightforward query.
</use_parallel_tool_calls>
```

**Evaluation approach:** Started with ~20 queries representing real usage. Early changes had dramatic impacts (prompt tweak: 30% → 80% success). LLM-as-judge evaluation scored outputs 0.0–1.0 against rubric criteria: factual accuracy, citation accuracy, completeness, source quality, tool efficiency. Human evaluation caught edge cases—early agents consistently chose SEO-optimized content farms over authoritative sources like academic PDFs.

**Production challenges:** Stateful error handling (can't restart on failure—too expensive; let the model adapt when tools fail, combined with deterministic safeguards like retry logic and checkpoints). Rainbow deployments for gradual version transitions. Non-deterministic debugging requiring full production tracing.

---

## 6. The Claude Agent SDK & Claude Code

### "Building agents with the Claude Agent SDK" — September 2025

*By Thariq Shihipar et al.*

**Plain-English summary:** The Claude Code SDK was renamed to the Claude Agent SDK to reflect that it powers far more than coding—deep research, video creation, note-taking, and other non-coding applications. The core design principle is that agents need the same tools programmers use: terminal access, file editing, file creation, and file search. By giving Claude a computer, it can read CSVs, search the web, build visualizations, interpret metrics, and perform all sorts of digital work.

**The agent loop:** `Gather context → Take action → Verify work → Repeat`

**Key capabilities:**
- **Agentic search:** Using bash scripts (`grep`, `tail`) to search through files; folder/file structure becomes context engineering
- **Semantic search:** Faster but less accurate—start with agentic search first, add semantic only if needed
- **Subagents:** Supported by default; enable parallelization and context isolation. Return only relevant excerpts, not full data
- **Compaction:** Auto-summarizes previous messages when context limit approaches
- **MCP integration:** Standardized integrations to external services (Slack, GitHub, Google Drive, Asana) with automatic authentication
- **Verification:** Rules-based feedback (linting), visual feedback (screenshots via Playwright MCP), LLM-as-judge

**Agent types enabled:** Finance agents, personal assistants, customer support, deep research agents.

**Resources:**
- SDK docs: https://docs.claude.com/en/api/agent-sdk/overview
- Custom tools: https://docs.claude.com/en/api/agent-sdk/custom-tools
- Subagents: https://docs.claude.com/en/api/agent-sdk/subagents
- MCP ecosystem: https://github.com/modelcontextprotocol/servers

---

### "Claude Code: Best practices for agentic coding" — April 2025

*By Boris Cherny et al.*

**Plain-English summary:** Claude Code is a command-line tool for agentic coding—intentionally low-level and unopinionated, providing close to raw model access. The article outlines six areas of best practices: customizing setup (CLAUDE.md files), giving Claude tools (bash, MCP, slash commands), common workflows (explore-plan-code-commit, TDD, visual iteration), optimization (specificity, images, context management), headless automation, and multi-Claude workflows.

**The CLAUDE.md pattern** is the most important configuration mechanism—a markdown file automatically pulled into context at every conversation start. Contents should include: bash commands, core files/utilities, code style, testing instructions, repo etiquette, and dev environment setup.

```markdown
# Bash commands
- npm run build: Build the project
- npm run typecheck: Run the typechecker

# Code style
- Use ES modules (import/export) syntax, not CommonJS (require)
- Destructure imports when possible (eg. import { foo } from 'bar')

# Workflow
- Be sure to typecheck when you're done making a series of code changes
- Prefer running single tests, and not the whole test suite, for performance
```

**Placement options:** Root of repo (most common, check into git), parent directories (monorepos), child directories (loaded on demand), home folder (`~/.claude/CLAUDE.md` for all sessions). Tune with emphasis like "IMPORTANT" or "YOU MUST" for better adherence.

**Custom slash commands** are stored as markdown in `.claude/commands/`:

```markdown
<!-- .claude/commands/fix-github-issue.md -->
Please analyze and fix the GitHub issue: $ARGUMENTS.

Follow these steps:
1. Use `gh issue view` to get the issue details
2. Understand the problem described in the issue
3. Search the codebase for relevant files
4. Implement the necessary changes to fix the issue
5. Write and run tests to verify the fix
6. Ensure code passes linting and type checking
7. Create a descriptive commit message
8. Push and create a PR
```

Usage: `/project:fix-github-issue 1234`

**Workflow patterns:**
- **Explore → Plan → Code → Commit:** Read files first (tell Claude NOT to write code yet), plan with extended thinking (trigger with "think", "think hard", "think harder", "ultrathink"), implement, commit
- **TDD:** Write tests → confirm they fail → commit tests → write implementation → commit code
- **Visual iteration:** Give Claude screenshot capability (Puppeteer MCP), provide a mock, iterate until matching
- **Safe YOLO mode:** `claude --dangerously-skip-permissions` in containers without internet

**Headless automation:**

```bash
# Non-interactive prompt execution
claude -p "<prompt>" --output-format stream-json

# Fan-out pattern for batch operations
for file in $(cat files.txt); do
  claude -p "Migrate $file from React to Vue." --allowedTools Edit Bash(git commit:*)
done

# Pipeline pattern
claude -p "<your prompt>" --json | your_command
```

**Multi-Claude workflows:** Writer + Reviewer pattern (separate terminals, separate contexts); multiple git checkouts for parallel tasks; git worktrees for lightweight parallel branches.

---

## 7. Evaluation Frameworks & Benchmarking

### "Demystifying evals for AI agents" — January 9, 2026

*Anthropic's most comprehensive evaluation guidance.*

**Plain-English summary:** This article provides a complete framework for building evaluations for AI agents, covering terminology, grader types, capability vs. regression evals, evaluation design for specific agent types (coding, conversational, research, computer use), and an 8-step roadmap. The key message: evals are your company's intellectual property and a crucial competitive advantage.

**Core terminology:**
- **Task/test case:** Single test with defined inputs and success criteria
- **Trial:** Each attempt (multiple needed due to non-determinism)
- **Grader:** Logic scoring performance (tasks can have multiple graders)
- **Transcript/trace:** Complete record of a trial
- **Evaluation suite:** Collection of related tasks

**Three types of graders:**

| Type | Methods | Best for |
|------|---------|----------|
| **Code-based** | String match, binary tests, static analysis, outcome verification, tool call verification | Fast, cheap, objective, reproducible |
| **Model-based** | Rubric scoring, NL assertions, pairwise comparison, multi-judge consensus | Flexible, captures nuance, scalable |
| **Human** | SME review, spot-check sampling, A/B testing | Gold standard quality, catches edge cases |

**Capability vs. regression evals:** Capability evals ask "what can this agent do well?" (start at low pass rate, give teams a hill to climb). Regression evals ask "does it still handle what it used to?" (should maintain ~100% pass rate). High-scoring capability evals "graduate" to regression suites.

**Example YAML for coding agent eval:**

```yaml
task:
  id: "fix-auth-bypass_1"
  graders:
    - type: deterministic_tests
      required: [test_empty_pw_rejected.py, test_null_pw_rejected.py]
    - type: llm_rubric
      rubric: prompts/code_quality.md
    - type: static_analysis
      commands: [ruff, mypy, bandit]
    - type: state_check
      expect:
        security_logs: {event_type: "auth_blocked"}
    - type: tool_calls
      required:
        - {tool: read_file, params: {path: "src/auth/*"}}
        - {tool: edit_file}
        - {tool: run_tests}
  tracked_metrics:
    - type: transcript
      metrics: [n_turns, n_toolcalls, n_total_tokens]
    - type: latency
      metrics: [time_to_first_token, output_tokens_per_sec, time_to_last_token]
```

**Non-determinism metrics:**
- **pass@k:** Probability of ≥1 success in k attempts (increases with k)
- **pass^k:** Probability all k trials succeed (decreases—e.g., 75% per-trial × 3 trials = 42%)

**8-step roadmap to great evals:**
1. Start early—20–50 tasks from real failures is sufficient
2. Start with manual checks—convert user-reported failures to test cases
3. Write unambiguous tasks with reference solutions
4. Build balanced problem sets—test both when behavior should AND shouldn't occur
5. Build robust eval harness—each trial starts from clean environment
6. Design graders thoughtfully—grade what agent produced, not the path; build in partial credit
7. Monitor for capability eval saturation—100% pass rate means the eval only tracks regressions
8. Keep suites healthy—dedicated eval teams own infrastructure; domain experts contribute tasks

---

### "Quantifying infrastructure noise in agentic coding evals" — November 2025

*By Gian Segato et al.*

**Plain-English summary:** Infrastructure configuration alone can swing agentic benchmark scores by several percentage points—sometimes exceeding the leaderboard gap between top models. A 6 percentage point gap (p < 0.01) was found between the most- and least-resourced setups on Terminal-Bench 2.0. Benchmark differences below 3 percentage points deserve skepticism.

**Key findings:** Two resource regimes exist. Up to ~3× the benchmark's specified resources, additional headroom fixes infrastructure reliability (error rates dropped from **5.8% to 2.1%**, p < 0.001) without making tasks easier. Beyond 3×, additional resources actively help agents solve harder problems—success rates climbed ~4 percentage points. The same pattern replicated on SWE-bench (**1.54 percentage points** at 5× RAM).

**Practical takeaway for analytics teams:** When evaluating model performance, ensure resource configuration is documented, consistent, and treated as a first-class experimental variable. Naive binomial confidence intervals already span 1–2 percentage points; infrastructure confounders stack on top.

---

### "Challenges in evaluating AI systems" — October 2023

*By Deep Ganguli, Nicholas Schiefer, Marina Favaro, Jack Clark.*

**Plain-English summary:** Even "simple" benchmarks have significant pitfalls. MMLU formatting changes alone cause **~5% accuracy swings**. BBQ (Bias Benchmark for QA) took one of Anthropic's best engineers a full uninterrupted week to implement—and initially, models scored 0 bias because they weren't answering questions at all. BIG-bench's 204 evaluations were too unwieldy to run efficiently. HELM doesn't use Claude's Human/Assistant format, giving misleading results.

**Six levels of evaluation difficulty:**
1. Multiple-choice benchmarks (MMLU, BBQ)—deceptively complex
2. Large benchmark suites (BIG-bench)—engineering-intensive, buggy
3. Curated expert frameworks (HELM)—no engineering effort but slow iteration, format mismatches
4. Crowdworker A/B tests—open-ended dialogue preference testing
5. Domain expert red teaming—national security threat assessment with clearances required
6. Third-party audits (Alignment Research Center)—independent dangerous capabilities assessment

---

## 8. Safety, Alignment & Interpretability Engineering

### "The engineering challenges of scaling interpretability" — June 2024

*Anthropic Interpretability Team.*

**Plain-English summary:** The team scaled from the "Towards Monosemanticity" paper (small transformer, October 2023) to "Scaling Monosemanticity" (Claude 3 Sonnet, May 2024), finding tens of millions of "features"—combinations of neurons relating to semantic concepts. Engineering is the major bottleneck to interpretability progress. Two detailed engineering problems are presented: distributed shuffling of 100TB+ training data, and building a feature visualization pipeline for millions of features.

**Distributed shuffle solution:** Each pass uses N jobs, each reading 1/N of the dataset, shuffling, and writing K files. The approach scales exponentially: with 100GB memory per job and 100 files per pass, 1 pass handles 100GB, 2 passes handle 10TB, 3 passes handle 1PB, 4 passes handle 100PB.

**Feature visualization pipeline:** Three-pass approach—(1) shard over dataset and features to find highest-activating tokens, (2) aggregate across shards, (3) compute surrounding context. An optimization added an intermediate pass sharded over the dataset to avoid random reads.

**Key research context:** Sparse Autoencoders (SAEs) are trained on transformer activations. Human evaluators found **70% of extracted features genuinely interpretable**. Safety-relevant features found include deception, sycophancy, bias, and dangerous content. Feature steering is "remarkably effective at modifying model outputs in specific, interpretable ways"—the famous Golden Gate Bridge demo clamped a feature at 10× maximum value, causing Claude to identify as the bridge.

### Constitutional AI

Constitutional AI (CAI) is Anthropic's approach to alignment. Two-phase training: (1) **Supervised phase**—the model critiques and revises its own responses using constitutional principles, then is fine-tuned on revised responses. (2) **RL phase**—the model evaluates response pairs, trains a preference model from AI preferences, then uses RL with the preference model as reward signal (RLAIF). Models >52B parameters were competitive with human feedback-trained preference models. Humans rate CAI models as more harmless than human-red-teamed models.

Anthropic also conducted **Collective Constitutional AI** where ~1,000 Americans drafted AI constitution principles via the Polis platform, finding areas of both agreement and disagreement with Anthropic's in-house constitution.

---

## 9. Inference, Deployment & Cost Optimization

### Prompt caching

Prompt caching reduces costs **up to 90%** and latency **up to 85%** for repeated context. The system stores a prompt prefix and reuses it on subsequent requests.

**Two cache tiers:**
- **5-minute TTL** (default): Free refresh on each hit; best for prompts used more frequently than every 5 minutes
- **1-hour TTL** (premium): 12× improvement; best for agentic workflows where follow-ups may take >5 minutes

**Implementation pattern:**

```python
import anthropic
client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system=[{
        "type": "text",
        "text": "You are an AI assistant tasked with analyzing literary works...",
        "cache_control": {"type": "ephemeral"}
    }],
    messages=[{"role": "user", "content": "Analyze the themes in..."}]
)
```

**Best practices:** Cache stable, reusable content (system instructions, background info, frequent tool definitions). Place cached content at the prompt's beginning. Up to **4 cache breakpoints** supported: tools, system instructions, RAG context, conversation history. Longer TTL entries must precede shorter ones.

**Pricing multipliers:** 5-minute cache writes are 1.25× base input price. Cache reads are 0.1× base price. 1-hour cache writes are higher but pay for themselves when hit rates are reasonable.

### Batch processing

50% discount on all tokens for non-time-sensitive workloads. Combine with 1-hour cache for better hit rates across batch requests. Ideal for bulk data analysis, large-scale classification, and periodic report generation.

### Latency optimization strategies

1. **Choose appropriate model:** Haiku 4.5 for speed-critical paths
2. **Minimize input/output tokens** while maintaining performance quality
3. **Use streaming** for responsive user experience
4. **Use prompt caching** for repeated context elements
5. Engineer the prompt first, THEN optimize for latency

### Cost architecture for analytics

The token economics fundamentally shape analytics system design. Multi-agent systems consume **~15× more tokens** than chat interactions. The Advanced Tool Use features reduce this dramatically: Tool Search Tool cuts 85% of tool definition tokens, Programmatic Tool Calling reduces intermediate results by 37%, and code execution with MCP achieves 98.7% reduction in specific workflows.

---

## 10. How AI Is Transforming Engineering Work

### "How AI is transforming work at Anthropic" — December 2025

*Research survey of 132 Anthropic engineers and researchers.*

**Plain-English summary:** Anthropic surveyed its own engineers in August 2025 and analyzed 200,000+ Claude Code usage transcripts. Engineers self-report using Claude in **59% of their work** (up from 28% a year ago) with a **50% productivity boost** (up from 20%). A **67% increase in merged PRs per engineer per day** was measured objectively when Claude Code was adopted org-wide. **27% of Claude-assisted work** consists of tasks that wouldn't have been done otherwise.

**Key findings for analytics teams:**
- Most common uses: debugging (55% daily), code understanding (42%), implementing new features (37%)
- Engineers delegate tasks that are: easily verifiable, low-stakes, boring, well-defined, outside core expertise but low complexity
- **14% are "power users"** reporting >100% productivity increase
- **8.6% of Claude Code tasks** are "papercut fixes"—minor quality-of-life improvements that would typically be deprioritized
- Engineers shifting to "70%+ being a code reviewer/reviser"
- Backend engineers building complex UIs they couldn't have done alone—the "full-stack" effect
- Claude Code autonomous actions before human input grew from ~10 to ~20 over 6 months

**Concerns identified:** The "paradox of supervision"—effectively using Claude requires supervision skills that may atrophy from overuse. One engineer reported using Claude for "80–90% of questions that used to go to colleagues," raising mentorship and knowledge-transfer concerns.

---

## 11. Practical Templates, Prompt Examples & Code Snippets

### System prompt template for analytics agents

Based on patterns from the context engineering and multi-agent articles:

```xml
<role>
You are a data analysis assistant specializing in {{DOMAIN}}.
You have access to tools for querying databases, creating visualizations,
and generating reports.
</role>

<instructions>
1. Always verify data quality before analysis
2. Use the appropriate tool for each data source
3. Present findings with specific numbers, not vague summaries
4. Cite the exact queries and data sources used
5. Flag any data anomalies or quality issues encountered
</instructions>

<tools_guidance>
- Use sql_query for structured database queries
- Use search_documents for unstructured text analysis
- Use create_visualization for charts and graphs
- Prefer specific, narrow queries over broad data pulls
</tools_guidance>

<output_format>
Structure all analysis reports with:
- Executive summary (2-3 sentences)
- Key findings (numbered, with supporting data)
- Methodology (queries used, data sources)
- Caveats and limitations
</output_format>
```

### CLAUDE.md template for analytics projects

```markdown
# Analytics Project Configuration

## Data Sources
- PostgreSQL: `psql -h analytics-db -U reader analytics`
- BigQuery: project `analytics-prod`, dataset `events`
- Redshift: Use `aws redshift-data` CLI

## Code Style
- Python 3.11+, use type hints
- Pandas for data manipulation, Polars for large datasets
- Plotly for interactive visualizations, Matplotlib for static
- All SQL queries must use parameterized inputs

## Testing
- pytest for unit tests: `pytest tests/ -v`
- Data validation: `python -m great_expectations checkpoint run`

## Workflow
- Always run type checking after changes: `mypy src/`
- Generate sample data for testing, never use production PII
- Document all assumptions in analysis notebooks
```

### SKILL.md template for analytics skill

```yaml
---
name: data-quality-audit
description: Audits datasets for completeness, consistency, and accuracy using statistical methods and automated checks
---

# Data Quality Audit Skill

## When to Use
Invoke this skill when asked to audit, validate, or assess the quality of a dataset.

## Process
1. Read the dataset schema and sample rows
2. Run completeness checks (null rates, missing values)
3. Run consistency checks (format validation, cross-field logic)
4. Run statistical outlier detection
5. Generate a quality scorecard

## Tools Available
- Run `python scripts/quality_check.py <filepath>` for automated checks
- See `reference/quality_metrics.md` for metric definitions
- See `templates/scorecard.md` for output format
```

### Evaluation task template (from evals article)

```yaml
task:
  id: "analytics-query-accuracy_1"
  description: "Generate correct SQL for a business question"
  input:
    question: "What was our MoM revenue growth in Q3 2025?"
    schema: "schemas/revenue.sql"
  graders:
    - type: deterministic_tests
      required: [test_query_returns_results.py, test_values_match_expected.py]
    - type: llm_rubric
      rubric: |
        Score the response on:
        1. SQL correctness (does it execute without errors?)
        2. Logic accuracy (does the query answer the actual question?)
        3. Efficiency (reasonable joins and filters?)
        4. Explanation quality (is the methodology clear?)
    - type: tool_calls
      required:
        - {tool: sql_query}
        - {tool: validate_results}
  tracked_metrics:
    - type: transcript
      metrics: [n_turns, n_toolcalls, n_total_tokens]
```

### Parallel tool use directive (from multi-agent research system)

```xml
<use_parallel_tool_calls>
For maximum efficiency, whenever you need to perform multiple independent
operations, invoke all relevant tools simultaneously rather than sequentially.
Call tools in parallel to run subagents at the same time. You MUST use parallel
tool calls for creating multiple subagents (typically running 3 subagents at
the same time) at the start of the research, unless it is a straightforward query.
</use_parallel_tool_calls>
```

### MCP server file tree generation (from code execution article)

```typescript
// ./servers/google-drive/getDocument.ts
import { callMCPTool } from "../../../client.js";

interface GetDocumentInput {
  documentId: string;
}
interface GetDocumentResponse {
  content: string;
}

export async function getDocument(
  input: GetDocumentInput
): Promise<GetDocumentResponse> {
  return callMCPTool<GetDocumentResponse>('google_drive__get_document', input);
}
```

### Context-aware agent prompt (from Claude 4 best practices)

```
Your context window will be automatically compacted as it approaches its limit,
allowing you to continue working indefinitely from where you left off. Therefore,
do not stop tasks early due to token budget concerns. As you approach your token
budget limit, save your current progress and state to memory before the context
window refreshes. Always be as persistent and autonomous as possible and complete
tasks fully, even if the end of your budget is approaching.
```

---

## 12. Master Reference Table of All Articles

| Date | Title | URL Slug | Primary Topic |
|------|-------|----------|---------------|
| Jan 9, 2026 | Demystifying evals for AI agents | `/engineering/demystifying-evals-for-ai-agents` | Evaluation frameworks |
| Dec 2, 2025 | How AI is transforming work at Anthropic | `/research/how-ai-is-transforming-work-at-anthropic` | Workforce impact |
| Nov 26, 2025 | Quantifying infrastructure noise in agentic coding evals | `/engineering/infrastructure-noise` | Benchmark methodology |
| Nov 24, 2025 | Advanced tool use | `/engineering/advanced-tool-use` | Tool Search, PTC, Examples |
| Nov 4, 2025 | Code execution with MCP | `/engineering/code-execution-with-mcp` | Token optimization |
| Oct 20, 2025 | Writing effective tools for AI agents—using AI agents | `/engineering/writing-tools-for-agents` | Tool design |
| Oct 16, 2025 | Equipping agents for the real world with Agent Skills | `/engineering/equipping-agents-for-the-real-world-with-agent-skills` | SKILL.md, composability |
| Sep 29, 2025 | Building agents with the Claude Agent SDK | `/engineering/building-agents-with-the-claude-agent-sdk` | Agent SDK |
| Sep 29, 2025 | Effective context engineering for AI agents | `/engineering/effective-context-engineering-for-ai-agents` | Context management |
| Apr 18, 2025 | Claude Code: Best practices for agentic coding | `/engineering/claude-code-best-practices` | Claude Code workflows |
| Jun 13, 2025 | How we built our multi-agent research system | `/engineering/multi-agent-research-system` | Multi-agent architecture |
| Dec 19, 2024 | Building effective agents | `/engineering/building-effective-agents` | Agent taxonomy |
| Oct 4, 2023 | Challenges in evaluating AI systems | `/research/evaluating-ai-systems` | Evaluation challenges |
| Jun 13, 2024 | The engineering challenges of scaling interpretability | `/research/engineering-challenges-interpretability` | Interpretability |

**Key open-source repositories referenced across articles:**
- Claude Agent SDK: https://docs.claude.com/en/api/agent-sdk/overview
- MCP protocol: https://modelcontextprotocol.io/
- MCP servers: https://github.com/modelcontextprotocol/servers
- Claude cookbooks: https://github.com/anthropics/claude-cookbooks
- Agent Skills standard: https://agentskills.io/
- Skills examples: https://github.com/anthropics/skills
- Computer use demo: https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo
- Claude Code: https://github.com/anthropics/claude-code
- Claude Code GitHub Action: https://github.com/anthropics/claude-code-action
- Patterns cookbook: https://platform.claude.com/cookbook/patterns-agents-basic-workflows
- Tool evaluation cookbook: https://platform.claude.com/cookbook/tool-evaluation-tool-evaluation
- Tool search cookbook: https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/tool_search_with_embeddings.ipynb
- PTC cookbook: https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/programmatic_tool_calling_ptc.ipynb

---

*This knowledge base was compiled on February 21, 2026 and reflects all articles published on Anthropic's engineering blog through January 2026, supplemented by relevant research articles and official documentation. Content should be validated against the latest documentation at docs.anthropic.com as Anthropic's platform evolves rapidly.*