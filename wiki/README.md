# Knowledge Base Wiki (Mobile-Accessible Mirror)

This directory contains a complete mirror of the [Knowledge Base Wiki](https://github.com/weisberg/knowledge_base_public/wiki) to provide access on mobile GitHub clients, which do not support wiki viewing.

## Structure

- **Main Pages**: All root-level `.md` files
- **Subdirectories**: Organized topic collections
  - `AI Agent Technologies/`: AI agents, agentic systems, and orchestration
  - `McKinsey AI Articles/`: McKinsey AI research and analysis
  - `CI Reference/`: Competitive intelligence resources
  - `Experimentation Notebook/`: A/B testing and experimentation methodology
  - `Public Discussion Research on 529 Plans_ A Report for Vanguard/`: 529 college savings plan research

## Navigation

Start with [`Home.md`](Home.md) for the complete table of contents and featured resources.

Key navigation files:
- **[Home.md](Home.md)**: Main index with all topics and resources
- **[_Sidebar.md](_Sidebar.md)**: Quick navigation sidebar content

## Featured Resources

### [Project Experiment](Project-Experiment.md)
A comprehensive, 25-lecture knowledge base on experimentation methodology, statistical techniques, and organizational implementation. 67 detailed documents covering statistical methods, organizational frameworks, regulatory compliance, and industry best practices from Netflix, Airbnb, Spotify, Microsoft, Amazon, and Vanguard.

### [Anthropic Engineering Knowledge Base](Anthropic-Engineering-Knowledge-Base.md)
The most comprehensive synthesis of Anthropic's engineering blog and research publications covering context engineering, agentic systems, tool use, MCP, evaluation frameworks, and deployment patterns.

### [Claude Code Architecture and Ecosystem: Exhaustive Technical Guide](The-Claude-Code-Architecture-and-Ecosystem-Exhaustive-Technical-Guide.md)
The definitive technical deep-dive into Claude Code's autonomous development platform, covering Sonnet 4.5 → 4.6 evolution, Model Context Protocol (MCP), programmatic tool calling, Agent Teams architecture, plugin system, and advanced orchestration patterns.

### [Vibe Analytics](Vibe-Analytics.md)
The future of data-driven decision making in the age of large language models and AI agents. Covers the paradigm shift from execution to orchestration, High-Fidelity Intent Specifications (HFIS), and Agile Agentic Analytics.

## Maintenance

The public KB lives in **two repos** and both must be updated for changes to appear everywhere:

1. **This mirror** (`knowledge_base_public/wiki/`) — pushed via the main repo's `origin/main`. Provides mobile access (GitHub mobile clients can't view wikis).
2. **The actual GitHub Wiki repo** (`knowledge_base_public.wiki.git`, branch `master`) — drives the wiki tab at https://github.com/weisberg/knowledge_base_public/wiki. Not cloned alongside the main repo.

### Sync workflow

When adding or editing a page:

1. Edit the file in this mirror directory.
2. Update `Home.md` and `_Sidebar.md` so the new page is linked.
3. Commit and push from the main repo (`origin/main`).
4. **Sync to the wiki repo** — pushing the mirror does NOT update the wiki tab:
   ```bash
   cd /tmp && rm -rf kb_wiki_sync
   git clone https://github.com/weisberg/knowledge_base_public.wiki.git kb_wiki_sync
   # copy changed files (new pages + Home.md + _Sidebar.md) into /tmp/kb_wiki_sync/
   cd /tmp/kb_wiki_sync && git add -A && git commit -m "..." && git push
   ```

The wiki repo is flat (no subdirectories). Filenames use dashes and become URL slugs.

---

*For the best experience with full wiki features (sidebar navigation, automatic linking), view the official wiki at: https://github.com/weisberg/knowledge_base_public/wiki*
