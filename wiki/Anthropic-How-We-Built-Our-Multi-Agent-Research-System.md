# How We Built Our Multi-Agent Research System

> **Source:** Engineering at Anthropic
> **Published:** June 13, 2025
> **Authors:** Jeremy Hadfield, Barry Zhang, Kenneth Lien, Florian Scholz, Jeremy Fox, and Daniel Ford
> **URL:** https://www.anthropic.com/engineering/built-multi-agent-research-system

---

*Our Research feature uses multiple Claude agents to explore complex topics more effectively. We share the engineering challenges and the lessons we learned from building this system.*

---

## Overview

Claude's Research feature allows it to search across the web, Google Workspace, and any integrations to accomplish complex tasks. The journey of this multi-agent system from prototype to production taught critical lessons about system architecture, tool design, and prompt engineering.

A multi-agent system consists of multiple agents (LLMs autonomously using tools in a loop) working together. The Research feature involves an agent that plans a research process based on user queries, and then uses tools to create parallel agents that search for information simultaneously. Systems with multiple agents introduce new challenges in agent coordination, evaluation, and reliability.

---

## Benefits of a Multi-Agent System

Research work involves open-ended problems where it's very difficult to predict the required steps in advance. You can't hardcode a fixed path for exploring complex topics, as the process is inherently dynamic and path-dependent. This unpredictability makes AI agents particularly well-suited for research tasks.

**The essence of search is compression:** distilling insights from a vast corpus. Subagents facilitate compression by operating in parallel with their own context windows, exploring different aspects of the question simultaneously before condensing the most important tokens for the lead research agent. Each subagent also provides separation of concerns -- distinct tools, prompts, and exploration trajectories -- which reduces path dependency and enables thorough, independent investigations.

### Key Performance Data

- A multi-agent system with Claude Opus 4 as the lead agent and Claude Sonnet 4 subagents **outperformed single-agent Claude Opus 4 by 90.2%** on internal research eval
- Three factors explained **95% of the performance variance** in the BrowseComp evaluation
- **Token usage alone explains 80%** of the variance, with number of tool calls and model choice as the other two factors
- Upgrading to Claude Sonnet 4 is a **larger performance gain than doubling the token budget** on Claude Sonnet 3.7

### When Multi-Agent Systems Excel

- Tasks involving heavy parallelization
- Information that exceeds single context windows
- Interfacing with numerous complex tools

### Trade-offs

- Agents typically use about **4x more tokens** than chat interactions
- Multi-agent systems use about **15x more tokens** than chats
- Not a good fit for tasks requiring shared context or many dependencies between agents
- Most coding tasks involve fewer truly parallelizable tasks than research

---

## Architecture Overview

The Research system uses a multi-agent architecture with an **orchestrator-worker pattern**, where a lead agent coordinates the process while delegating to specialized subagents that operate in parallel.

### System Components

1. **User** submits a query via Claude.ai chat
2. **System** creates a LeadResearcher agent
3. **LeadResearcher** enters an iterative research process:
   - Thinks through the approach and saves its plan to Memory
   - Creates specialized Subagents with specific research tasks
4. **Subagents** independently perform web searches, evaluate tool results using interleaved thinking, and return findings
5. **LeadResearcher** synthesizes results and decides whether more research is needed
6. **CitationAgent** processes documents and research report to identify specific locations for citations
7. Final research results with citations are returned to the user

### Key Architecture Decisions

- Dynamic multi-step search vs. traditional RAG (static retrieval)
- Memory persistence for plans (context windows exceeding 200,000 tokens get truncated)
- Separate CitationAgent for proper source attribution

---

## Prompt Engineering and Evaluations for Research Agents

Multi-agent systems have key differences from single-agent systems, including a rapid growth in coordination complexity. Early agents made errors like spawning 50 subagents for simple queries, scouring the web endlessly, and distracting each other with excessive updates.

### 8 Principles for Prompting Agents

1. **Think like your agents.** Build simulations using the Console with exact prompts and tools. Watch agents work step-by-step. This immediately reveals failure modes: agents continuing when they already had sufficient results, using overly verbose search queries, or selecting incorrect tools. Effective prompting relies on developing an accurate mental model of the agent.

2. **Teach the orchestrator how to delegate.** Each subagent needs an objective, an output format, guidance on tools and sources to use, and clear task boundaries. Without detailed task descriptions, agents duplicate work, leave gaps, or fail to find necessary information. Vague instructions like "research the semiconductor shortage" led to subagents misinterpreting tasks or performing the exact same searches.

3. **Scale effort to query complexity.** Embed scaling rules in prompts:
   - Simple fact-finding: 1 agent with 3-10 tool calls
   - Direct comparisons: 2-4 subagents with 10-15 calls each
   - Complex research: 10+ subagents with clearly divided responsibilities

4. **Tool design and selection are critical.** Agent-tool interfaces are as critical as human-computer interfaces. Give agents explicit heuristics: examine all available tools first, match tool usage to user intent, search the web for broad external exploration, prefer specialized tools over generic ones. Bad tool descriptions can send agents down completely wrong paths.

5. **Let agents improve themselves.** Claude 4 models can be excellent prompt engineers. When given a prompt and a failure mode, they can diagnose why the agent is failing and suggest improvements. A tool-testing agent that rewrites tool descriptions to avoid failures resulted in a **40% decrease in task completion time** for future agents.

6. **Start wide, then narrow down.** Mirror expert human research: explore the landscape before drilling into specifics. Prompt agents to start with short, broad queries, evaluate what's available, then progressively narrow focus.

7. **Guide the thinking process.** Extended thinking mode serves as a controllable scratchpad. The lead agent uses thinking to plan its approach, assess tools, determine query complexity, and define each subagent's role. Subagents use interleaved thinking after tool results to evaluate quality, identify gaps, and refine queries.

8. **Parallel tool calling transforms speed and performance.** Two kinds of parallelization introduced:
   - The lead agent spins up 3-5 subagents in parallel rather than serially
   - Subagents use 3+ tools in parallel
   - These changes **cut research time by up to 90%** for complex queries

### Prompting Strategy

Focus on instilling good heuristics rather than rigid rules. Strategies encoded in prompts include:
- Decomposing difficult questions into smaller tasks
- Carefully evaluating source quality
- Adjusting search approaches based on new information
- Recognizing when to focus on depth vs. breadth
- Setting explicit guardrails to prevent agents from spiraling out of control
- Maintaining a fast iteration loop with observability and test cases

---

## Effective Evaluation of Agents

Evaluating multi-agent systems presents unique challenges. Traditional evaluations assume deterministic paths, but multi-agent systems don't work this way -- agents might take completely different valid paths to reach their goal.

### Start Evaluating Immediately with Small Samples

In early agent development, changes tend to have dramatic impacts. A prompt tweak might boost success rates from 30% to 80%. Start with a set of about 20 queries representing real usage patterns.

### LLM-as-Judge Evaluation

LLM judge evaluates outputs against a rubric:
- **Factual accuracy** -- do claims match sources?
- **Citation accuracy** -- do cited sources match claims?
- **Completeness** -- are all requested aspects covered?
- **Source quality** -- did it use primary sources over lower-quality secondary sources?
- **Tool efficiency** -- did it use the right tools a reasonable number of times?

A single LLM call with a single prompt outputting scores from 0.0-1.0 and a pass-fail grade was the most consistent and aligned with human judgements.

### Human Evaluation Catches What Automation Misses

Human testers found that early agents consistently chose SEO-optimized content farms over authoritative but less highly-ranked sources like academic PDFs or personal blogs. Adding source quality heuristics to prompts helped resolve this.

### Emergent Behaviors

Multi-agent systems have emergent behaviors which arise without specific programming. Small changes to the lead agent can unpredictably change how subagents behave. The best prompts are not just strict instructions, but **frameworks for collaboration** that define division of labor, problem-solving approaches, and effort budgets.

---

## Production Reliability and Engineering Challenges

In agentic systems, minor changes cascade into large behavioral changes, making it remarkably difficult to write code for complex agents that must maintain state in a long-running process.

### Agents Are Stateful and Errors Compound

- Systems must durably execute code and handle errors along the way
- Built systems that can resume from where errors occurred
- Combine AI agent adaptability with deterministic safeguards like retry logic and regular checkpoints

### Debugging Benefits from New Approaches

- Agents are non-deterministic between runs, even with identical prompts
- Full production tracing lets teams diagnose why agents failed and fix issues systematically
- Monitor agent decision patterns and interaction structures without monitoring individual conversation contents

### Deployment Needs Careful Coordination

- Agent systems are highly stateful webs of prompts, tools, and execution logic
- Use **rainbow deployments** to avoid disrupting running agents, gradually shifting traffic from old to new versions

### Synchronous Execution Creates Bottlenecks

- Currently, lead agents execute subagents synchronously, waiting for each set to complete
- Asynchronous execution would enable additional parallelism but adds challenges in result coordination, state consistency, and error propagation

---

## Appendix: Additional Tips

### End-State Evaluation of Stateful Agents
Focus on end-state evaluation rather than turn-by-turn analysis. Evaluate whether the agent achieved the correct final state rather than whether it followed a specific process. For complex workflows, break evaluation into discrete checkpoints.

### Long-Horizon Conversation Management
Implement patterns where agents summarize completed work phases and store essential information in external memory before proceeding. When context limits approach, agents can spawn fresh subagents with clean contexts while maintaining continuity through careful handoffs.

### Subagent Output to a Filesystem
Direct subagent outputs can bypass the main coordinator for certain types of results. Implement artifact systems where specialized agents can create outputs that persist independently. This prevents information loss during multi-stage processing and reduces token overhead.

---

## Conclusion

When building AI agents, the last mile often becomes most of the journey. The compound nature of errors in agentic systems means that minor issues for traditional software can derail agents entirely. The gap between prototype and production is often wider than anticipated.

Despite these challenges, multi-agent systems have proven valuable for open-ended research tasks. Multi-agent research systems can operate reliably at scale with careful engineering, comprehensive testing, detail-oriented prompt and tool design, robust operational practices, and tight collaboration between research, product, and engineering teams.

---

*Written by Jeremy Hadfield, Barry Zhang, Kenneth Lien, Florian Scholz, Jeremy Fox, and Daniel Ford at Anthropic.*
