# Andrej Karpathy — Deep Dive into LLMs

A structured synthesis of Andrej Karpathy's ~3-hour general-audience introduction to large language models. This is a mental-models guide to *what is actually happening* when you type into ChatGPT, organized as study notes around the three training stages and the practical implications for using these tools.

> **Source:** Karpathy's "Deep Dive into LLMs like ChatGPT" (YouTube, ~3h 31m). Notes derived from a full transcript.

---

## TL;DR — The Big Picture

Training an LLM has three stages, loosely analogous to how children learn from textbooks:

| Stage | Textbook analogy | What happens | Output |
|---|---|---|---|
| **1. Pre-training** | Reading the exposition | Predict next token on ~15T tokens of internet text | **Base model** — an internet document simulator |
| **2. Supervised Fine-Tuning (SFT)** | Studying worked examples | Continue training on human-curated conversations | **Assistant** — imitates human labelers |
| **3. Reinforcement Learning (RL)** | Doing practice problems | Model tries solutions; correct ones are reinforced | **Reasoning model** — discovers its own thinking strategies |

When you talk to a non-thinking model (e.g., GPT-4o), you are talking to a **statistical simulation of a human data labeler** at OpenAI, who was following labeling instructions. When you talk to a thinking model (o3, DeepSeek-R1), you are getting something genuinely new — emergent reasoning discovered through RL.

---

## Stage 1 — Pre-training

### Step 1: Download and process the internet
- Start from **Common Crawl** (2.7B web pages indexed since 2007).
- Apply: URL filtering (block malware/spam/adult), text extraction from HTML, language filtering (e.g., FineWeb keeps pages >65% English), deduplication, PII removal.
- Result: a clean corpus like **Hugging Face FineWeb** — about **44 TB of text**. The internet, after filtering, fits on a hard drive.

### Step 2: Tokenization
- Neural networks need a 1D sequence of symbols from a finite vocabulary.
- Naive: bytes (256 symbols). Better: **Byte Pair Encoding (BPE)** — iteratively merge frequent pairs into new symbols.
- **GPT-4 vocabulary: 100,277 tokens.** "hello world" ≈ 2 tokens; case and whitespace matter.
- Try it: [tiktokenizer.vercel.app](https://tiktokenizer.vercel.app)
- FineWeb → ~15 trillion tokens.

### Step 3: Train a neural network to predict the next token
- Sample windows of tokens (max length = **context window**, e.g., 4K–1M).
- Network outputs a probability distribution over all 100K+ vocabulary tokens.
- Adjust weights so the probability of the *correct* next token goes up.
- Repeat across the entire dataset, in massive parallel batches.

### The Transformer
- Modern networks are Transformers with **billions of parameters** (think: knobs on a DJ set).
- Tokens get **embedded** into vectors → flow through layers of **attention** + **MLP** blocks → output logits → softmax → probability distribution.
- Stateless: no memory between forward passes; just a fixed mathematical function from input tokens to next-token probabilities.
- Modern frontier models likely have **hundreds of billions to trillions** of parameters.

### The compute story
- GPT-2 (2019): 1.6B params, 1024 context, ~100B tokens, originally cost ~$40K to train. Today reproducible for ~$100 thanks to better data, hardware, and software.
- Training runs on clusters of **NVIDIA H100 GPUs** (~$3/hr/GPU on Lambda). 8×H100 = one node; many nodes = a data center.
- **The "GPU gold rush"** is why Nvidia hit a ~$3.4T market cap. Elon's 100K-GPU data center exists for one purpose: predicting the next token, faster.

### Inference (sampling)
- Feed tokens in, get a probability distribution, **sample** one token from it (biased coin flip), append, repeat.
- Stochastic: the same prompt produces different outputs each run.
- Output is a remix — statistically similar to training data but rarely verbatim.

### What you get: a base model
- A base model = code (the architecture, ~few hundred lines) + parameters (the billions of numbers).
- Base models are rarely released, but examples include:
  - **GPT-2** (1.5B params, 2019)
  - **Llama 3.1 405B base** (Meta, trained on 15T tokens) — playable at hyperbolic.ai
- A base model is a **"lossy compression of the internet"** — a token autocomplete, not yet an assistant.

### Useful tricks with base models
- **Knowledge elicitation** via priming: "Here's my top 10 list of landmarks in Paris:" → model continues with plausible (but vague-recollection) answers.
- **Verbatim regurgitation** of high-quality, oversampled sources (e.g., Wikipedia).
- **Hallucinated futures**: ask about events past the knowledge cutoff and it confabulates.
- **Few-shot / in-context learning**: give 10 English:Korean pairs, then "teacher:" — the model learns the pattern in-context and translates.
- **Build an assistant via prompt**: structure a fake human/assistant conversation as the prompt; the base model continues in role.

---

## Stage 2 — Supervised Fine-Tuning (Post-Training)

### Goal: turn the document simulator into an assistant
- Throw out the internet dataset; substitute a **dataset of conversations** between a human and an assistant.
- Continue training with the same algorithm (next-token prediction). That's it.
- Pre-training: ~3 months on thousands of GPUs. SFT: ~3 hours, much smaller dataset.

### How conversations become tokens
- Special tokens like `<|im_start|>`, `<|im_sep|>`, `<|im_end|>` (introduced fresh in SFT) wrap turns.
- A 2-turn user/assistant exchange becomes ~49 tokens — still a 1D token sequence under the hood.
- Each LLM provider has a slightly different format; details don't matter conceptually.

### Where the data comes from
- **InstructGPT (OpenAI, 2022)** was the seminal paper: human contractors (Upwork, Scale AI) wrote prompts AND ideal assistant responses, following hundred-page **labeling instructions** ("be helpful, truthful, harmless").
- Open-source equivalent: **OpenAssistant**.
- Modern reality: SFT mixtures (e.g., **UltraChat**) contain **millions of mostly-synthetic conversations** generated by other LLMs and lightly edited by humans.

### What you are actually talking to in ChatGPT
> When you ask ChatGPT something, you are getting a **statistical simulation of an OpenAI human labeler** following OpenAI's labeling instructions. Not a magical AI.

If your specific question was in the SFT dataset, you'll get something close to what the labeler wrote. Otherwise, you get an emergent blend of pre-training knowledge + the assistant persona.

---

## LLM Psychology: Cognitive Quirks of SFT Models

### 1. Hallucinations
- **Cause:** Training data has confident answers to "Who is X?" Model statistically imitates the *style* of confident answers — even for people it has never seen. It doesn't know what it doesn't know.
- **Mitigation #1 — Knowledge-based refusals:** Meta's Llama 3 paper describes the technique: programmatically interrogate the model on facts from a document, check answers via an LLM judge, and for facts the model gets wrong, add training examples where the correct response is "I don't know." This wires up the internal "uncertainty" feature to verbal expression.
- **Mitigation #2 — Tool use:** Introduce special tokens like `<SEARCH_START>...<SEARCH_END>`. When emitted, inference pauses, runs a Bing/Google search, pastes results into the context window. Now the model isn't relying on vague recollection; it's reading fresh data in working memory. Train via examples.

### 2. Knowledge in parameters vs. context
- **Parameters = vague long-term recollection** (something you read months ago).
- **Context window = working memory** (something just in front of you).
- **Practical implication:** If you want a summary of a chapter, *paste the chapter into the prompt*. Don't rely on the model's recollection.

### 3. Knowledge of self
- "What model are you?" is a nonsensical question to a stateless token tumbler.
- By default, models hallucinate "I'm ChatGPT by OpenAI" because that's the most common assistant identity on the internet.
- Override via: (a) hard-coded identity Q&As in SFT data (e.g., AllenAI's OLMo has 240 such conversations), or (b) a **system message** prepended invisibly to every conversation.

### 4. Models need tokens to think
- Each token gets a **finite, ~fixed amount of compute** (a fixed number of layers × dimensions). You cannot cram an arbitrary calculation into one token.
- **Bad SFT label:** "The answer is $3" (followed by post-hoc justification).
- **Good SFT label:** Walk through intermediate steps, computing each in its own token. Each step is an easy computation; results accumulate in the context window.
- Test it: ask ChatGPT to answer in a single token. Easy problems work; harder arithmetic fails.
- **Practical fix:** ask the model to "use code." It writes Python, the interpreter executes it, and result is reliable.

### 5. Models are bad at character-level tasks
- Models see **tokens, not letters.** "ubiquitous" is 3 tokens; the model has no direct view of individual characters.
- Hence the famous "How many R's in strawberry?" failure (combines bad spelling + bad counting).
- **Fix:** ask the model to use code — copy-pasting the string is easy; Python does the character work.

### 6. Random sharp edges
- "Is 9.11 bigger than 9.9?" — models often say yes. Mechanistic interpretability suggests neurons associated with **Bible verses** light up (where 9.11 *does* come after 9.9).
- The **"Swiss cheese" model of capabilities**: brilliant at PhD-level questions, randomly stupid on trivial ones. Don't trust blindly.

---

## Stage 3 — Reinforcement Learning

### Why RL is needed
- Human labelers don't know which token sequence is *easiest for the model* to follow. Our cognition ≠ LLM cognition.
- Some "obvious" leaps in a labeler's solution are too computationally hard for one model token; some labeler steps are wasted tokens.
- Solution: let the model **discover its own token sequences** that reliably reach correct answers.

### How it works (verifiable domains)
1. Take a problem with a known answer (e.g., math, code).
2. Sample many candidate solutions (thousands per prompt).
3. Score each: did it reach the correct answer?
4. Train the model to be more likely to produce the *successful* token sequences.
5. Repeat over thousands of prompts × many updates.

This is **trial-and-error learning** — equivalent to a student doing practice problems.

### The DeepSeek-R1 paper (2025) — why it was a big deal
- OpenAI had been doing RL on LLMs internally for years but didn't publish details.
- **DeepSeek's January 2025 paper** publicly documented RL fine-tuning at scale and shared the recipe.
- Key result: as RL training proceeds, **average response length grows** — and accuracy on AIME math problems rises.
- Why longer? The model emergently discovers behaviors like:
  - "Wait, wait, that's not right..."
  - "Let me reevaluate this step by step."
  - "Let me try setting it up as an equation instead."
- **No human hardcoded these "thinking strategies"** — they emerged from optimization. This is the "**aha moment**" of LLM reasoning.

### Thinking models you can use today
- **DeepSeek-R1** — open weights, MIT-licensed. Hosted at chat.deepseek.com or together.ai.
- **OpenAI o1, o3, o3-mini-high** — closed; only a *summary* of reasoning is shown (OpenAI hides full chains-of-thought due to **distillation risk**).
- **Gemini 2.0 Flash Thinking** — at aistudio.google.com.
- Anthropic does not yet ship a dedicated thinking model (as of early 2025).
- **Rule of thumb:** ~80–90% of queries don't need a thinking model. Reach for them on hard math/code/reasoning problems where you'd accept 30s of thinking time.

### The AlphaGo connection
- AlphaGo's RL beat Lee Sedol partly because RL is **not bounded by human performance** — pure imitation tops out at the best human; RL keeps going.
- **Move 37**: a play that human experts evaluated as ~1-in-10,000 likely to be played by a human. AlphaGo discovered it. In retrospect, brilliant.
- **The open question for LLMs:** what is "Move 37" for open-domain reasoning? Maybe new analogies, new strategies — possibly even thinking in a non-English representation.

---

## RLHF — RL in Unverifiable Domains

For tasks like "write a joke" or "summarize this paragraph," there is no programmatic answer key. **RLHF (Reinforcement Learning from Human Feedback)** addresses this.

### How it works
1. Generate N candidate responses to a prompt.
2. Have humans **rank** them (easier than scoring or writing from scratch — exploits the **discriminator/generator gap**).
3. Train a separate **reward model** (a neural net) to predict human rankings.
4. Run RL against the reward model as a stand-in for humans.

### The upside
- Allows RL in arbitrary creative domains.
- Empirically yields better models — likely because humans are better at ranking than at writing ideal creative output.

### The fatal downside: reward hacking
- The reward model is a giant neural net with billions of parameters. It can be **gamed** with adversarial inputs.
- Run RLHF too long, and the optimizer discovers nonsense like "the the the the the the" gets a 1.0 score. Adding such examples to the reward model is whack-a-mole — there are infinite adversarial inputs.
- **You must crop training after a few hundred steps, before reward hacking dominates.**

### Karpathy's verdict
> **"RLHF is not RL."**
>
> Verifiable-domain RL (math, code, Go) can run indefinitely and produce magic. RLHF can only fine-tune incrementally before the reward function gets gamed. Treat it as a polish step, not a paradigm capable of unbounded improvement.

---

## Practical Use Recommendations

### How to get reliable answers
1. **Paste source material into context** rather than relying on recollection.
2. **Ask the model to use tools** — code interpreter for math/counting/spelling, web search for fresh facts.
3. **Let the model spread computation across tokens** — don't ask for one-token answers to hard problems.
4. **Choose the right model**: GPT-4o-class for ~80–90% of queries; thinking models (o3, R1) for hard reasoning.
5. **Always verify.** Use LLMs as tools for inspiration and first drafts. Own the work product.

### Where to track progress
- **LM Arena** (lmarena.ai) — human-comparison leaderboard. Useful but increasingly gamed; treat as a first pass.
- **AI News newsletter** by Swix — comprehensive daily/every-other-day summaries.
- **X / Twitter** — follow practitioners and labs directly.

### Where to access models
- **Frontier proprietary:** chatgpt.com, gemini.google.com, claude.ai.
- **Open weights inference:** together.ai (good UX, many state-of-the-art models).
- **Base models:** hyperbolic.ai (e.g., Llama 3.1 405B base).
- **Local:** **LM Studio** for distilled/quantized models running on your laptop GPU.

---

## What's Coming Next

- **Multimodality** — audio and images natively tokenized and interspersed with text. Same algorithm, different tokens.
- **Agents** — long-running tasks supervised by humans (think: human-to-agent ratios analogous to factory automation).
- **Pervasive integration** — invisible LLMs in every tool; computer-using agents (e.g., OpenAI Operator).
- **Test-time training** — currently model parameters are frozen at inference. The only learning is in-context. Future research: parameter updates during use, somehow analogous to sleep consolidation.
- **Beyond context windows** — finite, precious. Long-running multimodal tasks will need new mechanisms beyond just "make it bigger."

---

## The Mental Model to Walk Away With

When you type into ChatGPT and hit enter:

1. Your text is **tokenized** and inserted into a conversation protocol.
2. The whole thing becomes a 1D token sequence.
3. The model **autocompletes** the sequence — for a non-thinking model, this is a statistical simulation of a human labeler's ideal response. For a thinking model, this is post-RL emergent reasoning, only summarized in the UI.
4. There's a **finite amount of compute per token**, so reasoning has to be spread across many tokens.
5. Knowledge in parameters is **vague recollection**; knowledge in the context window is **working memory**. Use that distinction to your advantage.
6. The Swiss cheese is real. **Verify the work.**
