Executive Summary

Artificial intelligence (AI) has progressed from mastering games like chess and quiz shows in the 1990s

(Outside Scope) to competing with humans in complex business, economic, and strategic domains by 2015–

2025. This report surveys key studies where AI systems faced humans or acted autonomously in simulations

of operations, finance, strategy, policy, and negotiation. In domains with well-defined rules and abundant

data,   AI   agents   now   often  outperform   human   experts  (e.g.   strategy   games,   multi-player   poker,   data-

driven   business   decisions).   In   areas   requiring   open-ended   creativity,   long-term   judgment,   or   human

qualities (e.g. navigating unforeseen “black swan” events, empathic leadership, moral debates),  humans

still  hold  critical  advantages.  We  analyze   each   study   using   a   Q1–Q6   framework   –   problem  domain  &

analogy (Q1), AI design (Q2), performance vs humans (Q3), failure modes (Q4), transferability (Q5), and

sources   (Q6)   –   and   synthesize   trends.   The   trajectory   shows   AI   systems   rapidly   expanding   their
competencies:   from   perfect-information   board   games   (2016)   to   imperfect-information   and   multi-agent

games (2017–2019), and more recently to language-enabled strategy and business simulations (2022–2025).

Each generation of AI has closed more of the gap to human decision-making, leveraging advances in deep

learning   (especially   large   language   models,   LLMs),   self-play   reinforcement   learning,   and   multi-agent

simulation.

Overall,   AI’s   strength   lies   in   data-driven   optimization,   speed,   and   consistency,   while   humans   excel   at

adaptable reasoning, creativity, ethical judgment, and understanding other humans. We find AI already

dominates narrow strategic domains (Go, poker, video games), often discovering novel strategies beyond

human   intuition

1

2

.   In   complex   economic   simulations,   AI   can   design   effective   policies   or   business

strategies that match or beat human benchmarks under normal conditions

3

4

. However, today’s AI

agents  fail in long-horizon coherence  and  unpredictable scenarios: for example, even advanced LLM-

based agents eventually go off-track managing a simple vending business over many simulated days

5

,

and an AI “CEO” that excelled in routine decisions struggled with unforeseen shocks

4

. Transferability of

these AI solutions remains limited – systems trained for one game or scenario often require significant

adaptation to perform in another, though underlying techniques (self-play, two-level optimization, language
negotiation) provide a toolbox for new domains.

Looking ahead, enabling a “billionaire AI” agent – an autonomous system running a large enterprise or

investment   portfolio   at   top   human   (billionaire)   level   –  will   demand   further   breakthroughs.   Trends

supporting this include increasingly general and capable models (e.g. GPT-4 and successors), more faithful

simulations of real-world economies and markets for training, and hybrid human-AI decision frameworks

that leverage each’s strengths. At the same time, there are technical barriers (robust long-term planning,

common-sense reasoning, reliable alignment with human goals), economic and legal constraints (corporate

laws,   accountability,   investor   trust),   and   ethical   imperatives   (ensuring   AI-driven   businesses   don’t   cause

harm or inequality).  We estimate the probability of achieving a true “AI billionaire” agent within 10

years   to   be   low   (~20%),   but   rising   to   ~50%   within   20   years,   given   the   exponential   progress   in   AI

capabilities. It is also possible this goal is never fully realized under current paradigms (~10% chance), or

that   its   achievement   timeline   remains   fundamentally   uncertain.   In   any   case,   the   path   to   increasingly

autonomous AI in business is underway, demanding proactive consideration of safety, governance, and

human-AI collaboration to ensure beneficial outcomes.

1

Methodology

We   conducted   a   comprehensive   literature   scan   of   academic   and   industry   research   (2015–2025)   on   AI

systems   pitted   against   humans   or   operating   autonomously   in   complex   decision-making   simulations.

Sources included peer-reviewed journals and conference proceedings (Nature, Science, NeurIPS, ICML, AAAI,

IJCAI, AAMAS), pre-print servers (arXiv, SSRN), and white papers or blog posts from leading AI labs. We

identified seminal “grand challenge” studies (AlphaGo, Libratus, AlphaStar, OpenAI Five, etc.) and domain-

specific   experiments   in   business   strategy   and   economic   policy   (AI   Economist,   Cambridge   Judge   CEO

simulation, Andon Labs’ Vending-Bench). Each study was reviewed and dissected according to six questions:

Q1)  What   is   the   problem   domain   and   its   real-world   business/policy   analogy?  Q2)  What   AI   design/

architecture was used? Q3) How did the AI perform vs human participants or benchmarks? Q4) What failure

modes or weaknesses were observed? Q5) How transferable are the methods or insights to other domains?

Q6) Key citation(s). We favored sources that reported objective outcomes (win/loss records, profit metrics,

policy   efficiency)   and   included   commentary   on   limitations   or   implications.   In   synthesizing   results,   we

grouped studies by domain (operations, finance/economics, strategy, policy, negotiation) to compare where

AI excels versus where humans retain advantages. We then extrapolated trends to forecast AI’s trajectory

toward   autonomous   business   leadership,   and   analyzed   cross-disciplinary   barriers   (technical,   economic,

legal, ethical) and enablers. Any studies outside 2015–2025 are noted explicitly as “(Outside Scope)” for

context. We also flagged potential conflicts of interest or hype in sources via call-out notes when relevant.

This methodology ensured a balanced, up-to-date, and rigorous assessment of the state of AI-vs-human

performance in complex decision simulations.

Study Analyses (Chronological)

2016: DeepMind AlphaGo – AI Conquers Go

Q1 Domain & Analogy: Google DeepMind’s AlphaGo targeted the ancient board game Go, long seen as an

unsolved challenge for AI due to its enormous complexity. Strategically, Go is often likened to business or

military   strategy:   players   must   balance   tactical   skirmishes   with   long-term   territory   control.   AlphaGo’s

success hinted that AI could handle intricate, multi-step decision-making analogous to corporate strategy

contests or complex negotiations, where the search space of possibilities is vast and human intuition had

been paramount.

Q2 Design & Architecture: AlphaGo’s architecture combined deep neural networks (for pattern recognition

and   position   evaluation)   with   Monte   Carlo   tree   search   (for   lookahead   planning).   It   was   first   trained   on

human expert games  (supervised learning on 30 million moves) to imitate human Go knowledge, then

improved   by   playing   thousands   of   games   against   itself   (reinforcement   learning)

6

.   This   two-phase

training (initiate with human priors, then self-play to surpass human patterns) yielded a policy network to

select moves and a value network to judge win probabilities, all running on Google’s powerful TPUs.

Q3 Performance:  In March 2016, AlphaGo stunned the world by defeating Lee Sedol – one of Go’s top

champions with 18 international titles – in a 5-game match by 4–1

1

. This victory came a decade earlier

than   many   experts   predicted   humans   would   fall   in   Go.   AlphaGo’s   play   style   was   superhuman:   it   made

unconventional moves that initially baffled masters but proved extremely effective, earning it a 9-dan pro

ranking  and demonstrating that AI could uncover winning strategies humans hadn’t considered

7

. By

2

2017,   an   improved   version   beat   the   world   #1   (Ke   Jie)   3–0,   effectively   ending   human   dominance   in   Go.

AlphaGo’s Elo rating and consistency far exceeded any human, establishing a milestone: AI had overtaken

the very best human in one of the most complex board games.

Q4 Failure Modes:  AlphaGo did exhibit occasional weaknesses. Notably, in the one game it lost to Lee

Sedol, the human’s unexpected “hand of god” move (#78) confused AlphaGo, leading it to mis-evaluate the

game state and make a critical error

8

9

. This revealed that while AlphaGo was extremely strong, it

wasn’t infallible – it could falter when encountering patterns outside its training distribution. Its planning

was limited to the Go board; it had no broader context or adaptive common sense. Another limitation was

the compute intensity – AlphaGo required thousands of simulations per move and substantial hardware,

whereas a human uses intuition in real-time.

Q5 Transferability: The techniques behind AlphaGo (deep neural networks + tree search + self-play) proved

transferable  within  games   –   e.g.   DeepMind   soon   applied   the   same   template   to   chess   and   shogi   with

AlphaZero, achieving superhuman play in those games as well without human data

6

10

. These successes

suggest   AI   can   master   any   bounded,   rule-based   system   given   enough   training.   However,   transferring

AlphaGo’s approach to business or policy domains is non-trivial. Real markets don’t have perfect rules or a

single   clear   objective   like   “winning   a   game.”   Nonetheless,   AlphaGo   demonstrated   that   AI   can  learn

strategic reasoning: its ability to balance short-term and long-term tradeoffs and to surprise humans with

creative strategies gave hope that AI might assist in business strategy (e.g. finding non-intuitive solutions to

optimize   market   share   or   resource   allocation).   Direct   transfer   required   handling   uncertainty   and   multi-

agent interactions beyond two-player zero-sum settings.

Q6   Citation:  DeepMind’s   AlphaGo   was   documented   in  Nature  (Silver   et   al.,   2016)   and   made   headlines

worldwide. The SAAS Berkeley review summarizes the achievement: “In March 2016, AlphaGo stunned the Go

world by defeating 9-dan professional Lee Sedol 4–1…prior to this, many expected humans to retain an edge for

years.”

1

2017: CMU Libratus – Beating Poker Pros with Game Theory

Q1 Domain & Analogy: Libratus, developed at Carnegie Mellon University, tackled Heads-Up No-Limit Texas

Hold’em poker. Unlike Go, poker is an  imperfect-information  game: players have private cards and can

bluff or mislead. This domain is directly analogous to real-world strategic interactions under uncertainty –

such as business negotiations, contract bidding, cybersecurity face-offs, or military strategy – where each

party has hidden information and may use deception. Indeed, researchers explicitly note that many real

applications   can   be   modeled   as   imperfect-information   games   “such   as   negotiations,   business   strategy,

security interactions, and auctions”

11

. Libratus’s success would thus signify that AI can handle  strategic

reasoning with hidden information, a critical aspect of economics and policy-making.

Q2 Design & Architecture:  Libratus did not rely on deep neural networks but on computational game

theory and search. It featured a three-module architecture

12

13

: (1) an offline equilibrium solver that pre-

computed   an   approximate   Nash   equilibrium   strategy   for   an   abstracted   version   of   poker   (a   “blueprint”

strategy), (2) a  nested subgame solver  that, during play, handled specific situations in finer detail than the

blueprint   (re-solving   smaller   games   as   they   arose   when   new   information   was   revealed),   and   (3)   a  self-

improvement   module  that   analyzed   opponents’   play   to   patch   any   exploitable   weaknesses   in   its   strategy

between   sessions

14

15

.   These   algorithms   were   largely   domain-independent   and   rooted   in  recursive

reasoning and minimax equilibrium finding. Libratus ran on a high-performance supercomputing platform

3

to calculate its strategies. Importantly, it was not explicitly taught to bluff; bluffing behavior emerged from

the equilibrium computation – if a mix of bluffs maximized winning odds, Libratus would bluff.

Q3  Performance:  In  January  2017,  Libratus  made  history  by  defeating  a  team  of  four  top  professional

poker players in a 20-day “Brains vs AI” tournament of 120,000 hands

16

. The AI won decisively, finishing

with a  chip advantage of \$1,766,250  (in play money) over the humans – a result that was statistically

irrefutable as beyond luck

17

. This was the first time an AI beat elite humans in poker, a milestone in AI’s

ability to handle uncertainty and multi-step reasoning with hidden information. One pro player remarked

that Libratus was  “tougher than expected”  and kept improving throughout play

18

19

. Researchers noted

“the best AI’s ability to do strategic reasoning with imperfect information has now surpassed that of the best

humans”, calling it a new benchmark for AI

20

. Libratus’s win-rate was roughly 147 milli-big-blinds per hand

(mBB/hand), a very large margin

21

 – essentially it would win a significant amount of money on average

every hand, an unheard-of dominance at that level.

Q4  Failure   Modes:  Libratus  did  not  display   obvious  in-game  weaknesses   by   the   end   –  it   systematically

outmaneuvered the pros. Early in the event, the human team collaborated and attempted to find patterns

to exploit in Libratus’s strategy; whenever they identified a potential weakness, Libratus’s self-improvement

module   adjusted   and   closed   the   gap

22

.   This   resiliency   meant   Libratus   had   no   persistent   leaks   that

humans could abuse. However, Libratus had limitations in scope: it was specialized to two-player poker. The

techniques   had   to   abstract   the   game   (due   to  10^161  decision   points   possible

23

)   –   these   abstractions

could,   in   theory,   omit   some   nuance,   though   evidently   they   were   sufficient.   Outside   the   poker   domain,

Libratus lacked a natural ability to handle language or negotiate beyond betting patterns. And like AlphaGo,

it required considerable computing power (running on Pittsburgh Supercomputing Center) – an implicit

“failure” if one expects a human-like thinking process on commodity hardware.

Q5 Transferability: The significance of Libratus is its general approach to imperfect-information strategy.

The   algorithms   for   equilibrium   finding   and   subgame   resolution   are   not   poker-specific;   they   could   be

applied   to   any   competitive   scenario   with   hidden   information   and   well-defined   rules.   Potential   transfer

domains include: business negotiation (an AI agent negotiating contracts or prices), cybersecurity (planning

under uncertainty about an adversary’s state), military wargaming, or auctions/market bidding strategies.

In   fact,   the   CMU   team   pointed   out   applications   “in   any   realm   in   which   information   is   incomplete   and

opponents sow misinformation… Imagine your smartphone negotiating the best price on a new car for

you”

24

25

.   Such   prospects   show   transfer   to   real   economic   behavior   –   although   in   practice,   real

negotiations involve more complexity (e.g. human psychology, multi-issue tradeoffs) than two-player poker.

Libratus’s success directly paved the way for its multi-player successor Pluribus (2019) and influenced multi-

agent decision research. However, beyond games, deploying these methods requires modeling real-world

problems as games with objectives and utilities, which can be challenging. Also, equilibrium solutions can

be   computationally   expensive   to   find   as   complexity   grows.   Still,   Libratus   proved   AIs   can   achieve

superhuman negotiating tactics  in controlled settings, a step toward autonomous agents in business

strategy roles.

Q6 Citation: Brown & Sandholm presented Libratus’s methods at AAAI 2017 and in Science (2018). As CMU’s

news release summarized: “Libratus… defeated four of the world’s best professional poker players… The best AI’s

strategic   reasoning   with   imperfect   information   has   now   surpassed   the   best   humans.   This   milestone   has

implications for any realm with incomplete information and bluffing – business negotiation, military strategy,

cybersecurity…”

20

24

.

4

2017: DeepMind AlphaGo Zero – Self-Taught Mastery

Q1 Domain & Analogy: AlphaGo Zero was DeepMind’s next iteration after AlphaGo, focusing again on the

game of Go. However, its significance goes beyond Go – AlphaGo Zero demonstrated an AI that can teach
itself   superhuman   strategies   from   scratch.   In   a   business   context,   this   is   akin   to   an   AI   entrepreneur

learning   the   rules   of   a   new   market   and,   without   human   guidance,   eventually   outperforming   seasoned

experts. It showcases the potential of tabula rasa AI learning in any complex domain where the rules can be

simulated but optimal strategies are unknown.

Q2 Design & Architecture: AlphaGo Zero’s architecture was in some ways simpler, yet more advanced, than

AlphaGo’s. It used a single neural network (instead of separate policy and value networks) and  no human

training data whatsoever. The system started with only the basic rules of Go and learned entirely by playing

millions of games against itself (reinforcement learning with self-play). AlphaGo Zero introduced algorithmic

improvements  like   a   unified   Monte   Carlo   Tree   Search   that   consulted   the   neural   network   for   both   move

probabilities   and   value   estimates.   Crucially,   it   employed  structured   curriculum   learning  –   it   gradually

refined   its   play,   effectively   inventing   its   own   principles   of   Go.   This   approach   required   tremendous

computation   (though   notably  less  than   AlphaGo’s,   thanks   to   no   supervision   phase   and   a   smaller

architecture) and exemplified pure self-optimization.

Q3 Performance:  The results were astounding. In only  3 days  of self-play (4.9 million training games),

AlphaGo Zero surpassed all previous versions of AlphaGo. It decisively beat the 2016 champion-defeating

AlphaGo by  100 games to 0

10

. With just 72 hours of training, it achieved a level of play far beyond any

human or any AI that had learned from humans. Using a relatively modest number of TPUs, it not only

matched the earlier program but exceeded it – a Nature reviewer quipped “it wasn’t even worth humans

showing up” against this level of machine

10

. AlphaGo Zero attained an even higher Elo rating, and its style

was described as  alien yet effective. It rediscovered known strategies and then moved into totally new

territory,   innovating   sequences   that   humans   had   never   seen

26

27

.   By   2018,   the   same   algorithm

generalized into  AlphaZero, which mastered chess and shogi in hours, defeating the top human-trained

chess engine without any prior chess knowledge. This marked a paradigm shift: the strongest game players

on Earth were now AI agents that never needed human examples.

Q4 Failure Modes: AlphaGo Zero’s dominant performance left little room for in-game failure – no human or

earlier AI could challenge it in Go. However, one could point to its generalization limits: it was supremely

intelligent within the closed world of Go, but it had no understanding outside it. The system also required a

well-defined reward signal (winning/losing games) and enormous amounts of computation. If the rules or

objectives changed even slightly, one would have to retrain from scratch. Moreover, self-play in a purely

virtual setting means AlphaGo Zero had no exposure to out-of-game uncertainties – an analogy in business

would be an AI that excels in a simulated market but might be brittle if real-world factors intrude (e.g.

regulatory changes or human irrationality). In sum, while eliminating human bias from training, AlphaGo

Zero could still be limited by the simulation scope and training scale. There were also concerns about the

interpretability of its strategies – it’s so advanced that even Go experts struggle to understand some of its

moves, akin to an AI making counterintuitive business decisions that are hard to explain to humans (a

potential trust issue).

Q5 Transferability:  AlphaGo Zero’s success strongly validated  self-play reinforcement learning  as a tool for

any domain where an environment can be simulated. This has broad implications: in principle, an AI could

be dropped into a realistic business simulation or wargame and, given sufficient time and compute, learn to

5

outmaneuver   humans   without   being   taught   human   strategies.   The   two-level   learning   (starting   naive,

reaching superhuman) is very attractive for domains like logistics, finance, or industrial control – it might

discover   solutions   humans   never   devised.   However,   the   requirement   of   a   precise   simulation   and   clear

objective   is   a   bottleneck   for   transfer.   Unlike   games,   many   business   problems   don’t   have   a   simple   win

condition or may involve multiple objectives (profit, market share, customer satisfaction, etc.). Also, real

economic   environments   are   constantly   shifting,   whereas   Go’s   rules   are   static   –   an   AlphaGo   Zero   for

economics would need to continuously adapt. Nonetheless, elements of AlphaGo Zero’s approach (self-play,

no reliance on human data) have influenced subsequent AI Economist designs and multi-agent training

regimes. It represents the ideal of an autonomous strategist that creates its own knowledge. For achieving a

“billionaire AI,” this ability to self-learn new domains rapidly will be key, provided those domains can be

encoded in simulations.

Q6   Citation:  Silver   et   al.’s   2017   Nature   article   announced   AlphaGo   Zero.   A   University   of   Queensland

commentary highlighted: “Using less computing power and only three days of training, AlphaGo Zero beat the

original AlphaGo 100–0… AlphaGo Zero never saw humans play – it began knowing only the rules, then learned

superhuman performance.”

10

6

2019: OpenAI Five – AI Teamwork in Dota 2

Q1 Domain & Analogy: OpenAI Five took on Dota 2, a popular five-on-five multiplayer online battle arena

(MOBA) game. Dota 2 is a real-time strategy game requiring teamwork, quick reactions, long-term resource

management,   and   imperfect   information   (fog-of-war   hides   opponents).   This   domain   is   analogous   to

running a  competitive business team or military squad: success requires coordinating multiple agents

(heroes  in  the  game,  or  departments  in   a  company),   responding   to   opponents’   strategies,   and   making

hundreds of local decisions that feed into an overall victory. Dota 2 is an open-ended environment – there

are many ways to win and the game state is partially observable – making it an ideal test of AI’s ability to

handle   complexity   and   collaboration.   For   business,   OpenAI   Five’s   multi-agent   coordination   hints   at   AI

managing teams or processes in dynamic markets.

Q2 Design & Architecture:  OpenAI Five consisted of  five neural network agents, one controlling each

hero on a team, trained together via deep reinforcement learning. The training used self-play at massive

scale:   the   AI   team   played   millions   of   games   against   itself   (and   past   versions)   using   a   scaled-up   policy

gradient method. To stabilize learning in such a complex game, OpenAI engineered a league system of bots

to prevent overfitting to any single strategy. They also shaped rewards for key objectives (kills, objectives

taken) to guide the agents. Importantly, OpenAI Five had no built-in knowledge of Dota 2’s strategies – it

learned from scratch by trial and error. Over 10 months, the AI accumulated  ~45,000 years of gameplay

experience  in equivalent – an enormous training investment

28

29

. The architecture included an LSTM-

based network for each hero to handle partial observability and long-term dependencies (like remembering

what happened across a 40-minute game). The agents communicated implicitly by observing each other’s

actions and the shared game state, rather than explicit messaging.

Q3 Performance: By April 2019, OpenAI Five reached a historic milestone: it defeated the reigning Dota 2

world champion team OG in a public match, winning 2 games in a row

30

. This was the first time an AI

had beaten  top professional human players in a full e-sport  with standard rules. The victory was decisive –

OpenAI   Five   exhibited   superior   teamfight   coordination,   resource   allocation,   and   tactical   coherence.   It

surprised   commentators   with   its   unconventional   strategies   (e.g.   highly   aggressive   plays,   unusual   item

choices) executed with machine precision. In terms of metrics, the AI achieved a >99% win-rate against

6

amateur   and   semi-pro   teams   during   testing,   and   against   OG   (the   world’s   best   at   the   time)   it   won

convincingly.   Its   playstyle   was   described   as   at   times   “alien”   but   fundamentally   sound   and   relentlessly

efficient. The OpenAI team also measured that the bots could play at an equivalent  Actions-Per-Minute

(APM)  and   reaction   speed   comparable   to   humans,   after   enforcing   limits   (to   avoid   unfair   superhuman

clicking   speed).   OpenAI   Five’s   achievement   demonstrated   that   AI   can   handle  real-time,   team-based

strategy at the highest level, a task far more complex than previous turn-based or one-on-one games.

Q4 Failure Modes:  OpenAI Five did have constraints and weaknesses. First, the AI’s understanding was

limited to the distribution of games it trained on. It played with a restricted pool of 17 heroes (out of over

100 in Dota 2) to keep the learning problem tractable. Humans might exploit out-of-distribution scenarios

or heroes (though in the controlled match, those were disallowed). When OpenAI Five was later opened to

the public in an online event, some amateur players found occasional quirky behaviors to exploit – for

instance, the AI sometimes prioritized objectives in a predictable way that creative human tactics could

counter. Additionally, OpenAI Five lacked long-term planning beyond a single match – it had no concept

of a tournament meta-strategy or adapting over multiple games except via retraining. Its coordination,

while excellent, was purely learned and not easily interpretable; if something went wrong, it couldn’t explain
its decisions. Another failure mode was the cost and time of training: requiring tens of thousands of years

of experience is impractical outside of simulation. This highlights the challenge if one tried to train a similar

AI  on  real-world  business  data  (you  can’t   run   45,000   years   of   a   company’s   life  in   practice).   In   essence,

OpenAI Five could occasionally be  exploited by strategies it hadn’t seen, and it wasn’t general beyond

Dota 2’s rules.

Q5   Transferability:  OpenAI   Five’s   success   underscored   that   multi-agent   RL   can   produce   emergent

teamwork and strategy. The techniques (self-play league training, reward shaping, large-scale simulation)

are   transferable   to   other   multi-agent   scenarios   –   potentially  robotics   swarms,  cooperative   business

processes,   or  simulated   economies.   For   example,   managing   a   fleet   of   autonomous   vehicles   or

coordinating supply chain agents could leverage similar reinforcement learning of teamwork. Moreover,

OpenAI Five showed an AI can learn  cooperative behavior  (like dividing roles, assisting allies) which is

crucial in real organizations. However, transferring to real business requires a reliable simulator or digital

twin of the environment; that’s feasible in some cases (e.g. warehouse logistics simulators) but not for all

(e.g. unpredictable human-driven markets). Nonetheless, the idea of an AI that can rapidly play through

scenarios and learn optimal team decisions is very powerful for strategy and operations research. OpenAI

Five’s evolution also suggests that if one could model, say, a corporate war-game between companies, an AI

might   self-play   it   to   discover   innovative   tactics.   In   practice,   human   oversight   and   validation   would   be

needed – direct transfer to real operations has safety and reliability hurdles.

Q6 Citation: OpenAI documented this in an April 2019 press release: “OpenAI Five is the first AI to beat the

world champions in an esports game, having won two back-to-back games versus the world champion Dota 2

team (OG)… the first time an AI has beaten esports pros on livestream.”

30

2019: DeepMind AlphaStar – Grandmaster in StarCraft II

Q1 Domain & Analogy: AlphaStar, from DeepMind, targeted StarCraft II, another hugely popular real-time

strategy (RTS) game. StarCraft II involves managing an economy, building armies, and engaging in battles

in real-time, under partial information. It’s often likened to commanding a war or managing a complex

operation: players must make split-second tactical decisions while executing a broader strategy of resource

management and tech development. In business terms, StarCraft II could be seen as analogous to running

7

a   company   in   a   competitive   market   –   balancing   R&D,   resource   allocation,   and   direct   competition

simultaneously.   AlphaStar’s   domain   required   mastering  strategic   planning,   real-time   control,   and

adaptive strategy, all of which are directly relevant to high-level business simulations (like operational

wargames or crisis response scenarios).

Q2   Design   &   Architecture:  AlphaStar’s   architecture   was   an   advanced  multi-agent   reinforcement

learning system. It began with supervised learning on human game replays (to bootstrap basic strategies

and mimic human play patterns)

31

. Then, like OpenAI Five, it used a self-play league – a population of AI

agents that played against each other (and past versions) to improve. This league included agents with

different   strategies   to   ensure   robustness   (some   agents   intentionally   focused   on   certain   tactics   to   force

others to adapt). AlphaStar’s neural network processed raw game data (the units and features visible to it)

and output actions like unit commands. A key component was the use of an LSTM-based core that allowed

planning over time and dealing with partial observability. DeepMind also imposed human-like constraints:

AlphaStar’s agents had a limited camera view of the map and an action rate limit to mimic human physical

constraints

32

33

. This was important to ensure the AI’s win wasn’t simply due to superhuman clicking or

seeing the whole map at once. Training AlphaStar was extremely computationally intensive – reportedly
involving hundreds of TPUs over many days, with the final league producing agents that had experienced

the equivalent of years of gameplay (though less publicly quantified than OpenAI Five’s training). The final

AlphaStar actually consisted of separate specialist agents for each of the game’s three factions (Protoss,

Terran, Zerg), each a deep network that learned to master that race

34

.

Q3 Performance: In 2019, AlphaStar achieved Grandmaster level on the official StarCraft II ladder for all

three races, meaning it ranked above 99.8% of human players in the world

33

. This was verified under

normal playing conditions on Battle.net (the online platform), making AlphaStar essentially one of the top

few   dozen   players   globally.   Earlier   in   January   2019,   prototype   AlphaStar   agents   had   privately   defeated

professional   players   in   Protoss   vs   Protoss   matches   (with   some   conditions),   though   in   a   subsequent

exhibition a human pro managed to win a game when the AI was constrained with a new camera view –

showing the importance of those fairness constraints. By the October 2019 result, AlphaStar had adapted

and proven itself across all races and maps. The achievement is monumental: StarCraft II is an open-ended,

long-horizon game with high stochasticity and complexity. AlphaStar’s play was noted for its precision and

also   some   distinctly   non-human   strategies.   For   example,   it   executed   multi-front   attacks   flawlessly   and

showed no fatigue or attention lapses, which often gave it an edge in late-game situations where humans

get overwhelmed. Its strategic understanding – timing attacks, countering opponent’s build orders – was at

pro level, and it even developed unexpected tactics (one anecdote: it made use of units or combinations in

proportions rarely seen in human play). Overall, AlphaStar demonstrated that an AI can reach elite human

performance   in   a  complex,   semi-real-time   environment   with   large   action   space  –   a   significant   step

beyond prior turn-based or simpler games.

Q4 Failure Modes:  While AlphaStar is extremely strong, it had some vulnerabilities and limitations. Early

versions were exploitable with specific “cheese” strategies – for instance, humans found that the AI was

weak against tactics it hadn’t trained much against, like certain all-in rushes or unusual base placements

(one pro beat an earlier AlphaStar by hiding a building in a far corner of the map, which the AI failed to

scout adequately). The final version became more robust via the league training covering many scenarios,

but in principle, any fixed AI policy in a game this complex could have blind spots. Another limitation is

adaptability during a match series: human champions often adapt their strategy between games in a

best-of series, whereas AlphaStar did not have a mechanism to carry over learning or counterplanning

between ladder games. Additionally, AlphaStar’s style, optimized for winning, might sometimes appear alien

8

– for example, it might execute strategies that exploit the game mechanics in non-intuitive ways that a

human   wouldn’t   (though   not   illegal   in   game   terms,   sometimes   it   micromanaged   units   with   inhuman

efficiency, raising questions of fairness until the action limits were enforced). Technically, like others, it was a

black   box   neural   network   –   it   might   win   without   explanation,   which   could   be   a   problem   in   critical

applications. Outside of StarCraft, the strategy would not directly generalize; one would need to retrain in

any new domain from scratch. AlphaStar also consumed enormous resources to train (tens of millions of

game steps, etc.), reflecting diminishing returns – to push a bit higher, exponentially more compute was

thrown in. This suggests current methods scale expensively.

Q5 Transferability: AlphaStar’s success reinforces the trend that  multi-agent learning and self-play can

solve extremely hard problems. The general methods (neural network function approximation, imitation +

reinforcement learning, league training) could be applied to other competitive environments. For example,

one could imagine an AlphaStar-like system in a macroeconomic simulation where agents (companies or

countries) compete and cooperate – the system could potentially learn emergent strategies for trade or

conflict.   Likewise,   in   robotics   or   fleet   management,   a   similar   approach   might   optimize   multi-robot

coordination. AlphaStar also integrates imperfect information and real-time planning, which are central
to many real-world tasks (financial trading under uncertainty, managing a power grid in real-time, etc.). The

fact that it had to handle incomplete knowledge and still perform optimally is a big transferable insight: AI

can do strategic planning without full information, using prediction and inference. On the other hand, direct

transfer is limited because real business or policy environments might not have as clean a reward signal as

a   game’s   win/loss,   and   they   may   involve   continuous   adaptation   to   changing   rules.   Nonetheless,   the

techniques   behind   AlphaStar   suggest   that   if   we   can   simulate   something,   we   might   train   an   AI   to

superhuman   performance   on   it   –   a   hopeful   sign   for   complex   simulations   of   economies   or   industries.

DeepMind itself suggested these methods “could be applied to many other domains” beyond StarCraft

35

.

AlphaStar represents a blueprint for autonomous strategy agents operating in chaotic environments – a

stepping stone to AI agents that could manage real-time business operations or military tactics.

Q6 Citation: DeepMind announced the result in late 2019: “Using the advances described in our Nature paper,

AlphaStar was ranked above 99.8% of active players on Battle.net, achieving Grandmaster level for all three races

in StarCraft II.”

33

2019: Facebook/CMU Pluribus – Mastering Multiplayer Poker

Q1 Domain & Analogy: Pluribus extended AI poker from the two-player case to six-player no-limit Texas

Hold’em, the most common form of poker. Multi-party poker is a much harder strategy problem because of

more complex hidden information interactions and the absence of a single opponent to target – it’s a free-

for-all of shifting advantages. This domain corresponds to many real-world strategic settings with multiple

self-interested   parties,   such   as   financial   markets   with   many   traders,   auctions   with   several   bidders,   or

multi-company negotiations. In these scenarios, an agent must handle bluffing and alliances in a more

fluid, less predictable environment than head-to-head games. Achieving superhuman performance in six-

player poker signified that AI can thrive in  competitive many-agent environments  analogous to open

markets or complex negotiations with several stakeholders.

Q2   Design   &   Architecture:  Pluribus   built   on   techniques   from   Libratus   but   introduced   innovations   for

tractability in multiplayer. The AI used a combination of  self-play reinforcement learning  (to develop a

base strategy) and real-time search. Key to Pluribus was a strategy called  “blueprint + real-time subgame

solving”: it first computed an approximate equilibrium strategy for the whole game using self-play (storing

9

this strategy as a blueprint policy), then during actual play, it looked a few moves ahead (searching a limited

game tree) whenever it reached a decision, tailoring its play to the specific situation. Unlike Libratus, which

had to adjust to one opponent, Pluribus had to account for five other players. It did so by simulating likely

continuation strategies of all players (including copies of itself for opponents) and finding a robust decision.

The AI ran on comparatively modest cloud servers (not a full supercomputer) and could play multiple hands

in  parallel  to  speed  up  learning.  Also  notable,  Pluribus  did  not  use  deep  neural  networks  for  decision-

making;   it   represented   the   poker   game   states   with   handcrafted   features   and   used   faster   planning

algorithms,   which   was   feasible   due   to   domain   expertise   and   the   smaller   action   space   (bet   sizes   were

discretized). Its training was unsupervised self-play – playing against copies of itself to refine its strategy

iteratively, very similar to how Libratus self-improved.

Q3 Performance:  In 2019, Pluribus became the first AI to beat top human professionals in multiplayer

poker. In a series of experiments, Pluribus faced off against elite poker pros (including World Poker Tour

champions). One test had one AI vs five human pros playing 10,000 hands – Pluribus “emerged victorious”

with a statistically significant win margin

36

37

. Another experiment had five instances of Pluribus at a

table with one human (to gather more data per human) – again the AI consistently won. To quantify the
performance:  researchers  noted  Pluribus’s  win  rate  was  about  \$1,000  per  hour  against  these  pros  (in

theoretical   currency)

38

.   This   win   rate   is   substantial   at   those   stakes   and   corresponds   to   the   AI   taking

roughly 5 big blinds per 100 hands from the opposition, which is a crushing margin against world-class

players. Tuomas Sandholm summarized: “Pluribus achieved superhuman performance at multi-player poker, a

recognized milestone in AI and game theory”

39

. Not only did it win, but it did so by unexpected strategies:

for example, Pluribus used an unconventional play known as “donk betting” (leading into the bettor) far

more often than humans, turning a traditionally “weak” move into part of an optimal strategy

40

. The

humans were impressed; one pro noted the AI’s major strength was its ability to use mixed (randomized)

strategies   perfectly,   something   humans   struggle   to   execute   consistently

41

.   Experts   believe   some   of

Pluribus’s strategies  “might even change the way pros play the game”

42

, highlighting how AI can discover

superior tactics.

Q4 Failure Modes: Pluribus was very strong, but as with Libratus, its strategies are equilibrium-centric and

can be counter-intuitive. If opponents did not play in the realm of its Nash equilibrium approximation, could

they exploit the AI? The research suggests Pluribus is strongly robust – by computing a balanced strategy, it

prevents systematic exploitation. The human pros did try to find weaknesses during play, but Pluribus’s

unpredictability   (due   to   its   mixed   strategies)   made   it   hard   to   pin   down   a   clear   counter.   However,   one

limitation   is   computational:   Pluribus’s   real-time   search   had   to   cut   off   after   a   limited   depth   due   to

exponential branching with six players; it’s possible a highly unorthodox sequence of bets not anticipated

by the blueprint could confuse it. There were no reports of humans finding such a sequence in the trials,

but in principle multi-player game theory is not as well understood as two-player, so Pluribus’s strategy

might not be a true equilibrium. Another failure mode is generality – Pluribus knows poker, nothing else. It

cannot, for instance, explain why it made a bet in plain language or carry those negotiation skills into, say, a

settlement   discussion.   Also,   Pluribus   still   had   to   simplify   poker   (e.g.   limiting   some   bet   sizes   in   its

abstraction), which a clever human might exploit by doing something outside that abstraction. In these

experiments, stakes and rules were controlled to standardize play. In real-world multi-party negotiations,

the “rules” are far messier.

Q5   Transferability:  Pluribus   underscores   that   AI   can   handle  multi-agent   interactions  with   private

information. This is directly transferable to scenarios like auctions (multiple bidders with private valuations),

negotiations (several parties negotiating a deal, possibly making offers and counteroffers akin to bets), and

10

financial markets (many traders bluffing or signaling with trades). The researchers explicitly pointed out

that the techniques could address “a wide variety of real-world problems” where, “like in poker, [they] involve

actors   who   bluff   or   hide   key   information.”

43

44

.   This   could   include   cybersecurity   (multiple   attackers/

defenders), political negotiations (multiple countries with secret agendas), or complex business strategy

where  several  firms  interact.  The  AI  approach  –  computing  a  principled  strategy  and  then  adjusting  to

actual moves – could be a template for AI advisors in these domains. However, applying it outside games

requires formalizing those problems in game-theoretic terms and ensuring the AI’s actions translate to real

action   (e.g.   making   a   contract   offer).   Pluribus   also   demonstrated   how  action   randomization  can   be   a

powerful   tool   (to   prevent   predictability),   which   is   a   strategy   humans   use   in,   say,   pricing   strategies   or

bargaining (not revealing a consistent pattern). AIs could implement mixed strategies more rigorously than

humans, potentially giving them an edge in competitive business. We should note, however, that real multi-

party interactions often include communication, reputation effects, emotional factors – none of which were

present in silent poker. So transferring Pluribus’s poker prowess to, say, multi-party contract negotiations

would require integrating language and adaptive modeling of human behavior, which is something later

agents like Meta’s Cicero (for Diplomacy) begin to tackle.

Q6   Citation:  Pluribus’s   success   was   reported   in  Science  (Brown   et   al.,   2019)   and   summarized   by   NPR:

“Researchers… designed a bot called Pluribus capable of taking on poker professionals in the most popular form

of poker and winning… If this were for real money, the bot would be winning at a rate of about \$1,000 an

hour.”

38

.   An   AFP/Phys.org   piece   added,  “Thus   far,   superhuman   AI   milestones   in   strategic   reasoning   were

limited to two-player games. Pluribus achieving superhuman performance in six-player poker is a breakthrough.

The creators say this could apply to real-world problems with bluffing or hidden information.”

39

44

.

2019: IBM Project Debater – Man vs Machine in Rhetoric

Q1 Domain & Analogy: Project Debater  by IBM Research is a unique entrant: an AI system designed to

engage   in  competitive   debates   with   human   experts  on   complex   topics.   Unlike   games   or   numeric

simulations, debating is an open-ended task requiring language, knowledge, reasoning, and persuasion.

The domain here is formal debate (with timed speeches for and against a proposition), which serves as an

analogy for persuasive communication and negotiation in business and policy. In business, an executive

must argue for a strategy; in policy, leaders debate legislation. If an AI can debate, it hints at AI’s ability to

autonomously   participate   in   discussions,   make   cases,   and   influence   decisions   –   a   step   toward   AI   in

boardrooms  or  courtrooms.  Project  Debater’s  scope  included  parsing  a  topic  (like  “should  we  subsidize

space exploration”), marshalling arguments and evidence, delivering speeches, and rebutting an opponent.

This is analogous to an AI CEO or advisor formulating strategy pitches or policy arguments on the fly.

Q2   Design   &   Architecture:  Project   Debater’s   architecture   combined   several   AI   components.   It   had   a

massive corpus of newspaper and journal text (hundreds of millions of sentences) and used an argument

mining  module to retrieve relevant points for a given debate topic. A core  NLP engine  then constructed

coherent arguments and counterarguments from these points. For speech delivery, it used advanced text-

to-speech.  Importantly,  Debater  had  to  perform  listening  comprehension:  when  the  human  opponent

spoke, Debater transcribed the speech, identified key claims, and generated rebuttals. The design did not

rely on machine learning for the overall debate strategy – rather it was a collection of hand-crafted modules

(for claim detection, evidence retrieval, narrative generation) informed by AI algorithms (some ML, some

rule-based). IBM described it as a “hybrid” AI – combining data-driven learning with knowledge bases

and   language   understanding.   There   was   no   reinforcement   learning   of   debating;   instead,   developers

evaluated it on many topics to refine its performance. The architecture also included a  argumentative

11

strategy  layer   to   prioritize   which   points   to   use   (for   instance,   prefer   stronger   evidence   and   diverse

arguments). Essentially, Project Debater was an orchestrator of language technologies geared toward one

goal: make a persuasive, well-structured case on any given topic.

Q3   Performance:  Project   Debater’s   public   debut   came   in   June   2018   and   February   2019   in   a   series   of

exhibition debates against top human debaters (including a world debate champion, Harish Natarajan). In

terms of pure win/loss by audience vote, Debater narrowly lost the main debate in 2019 – the audience

judged the human to be more persuasive overall. However, the AI held its own impressively. It delivered a 4-

minute opening argument with substantial points and a rebuttal to the human’s argument, followed by a

closing summary. Observers noted that Debater was  better than humans at bringing in relevant facts

and   statistics,   drawing   from   its   vast   corpus   (e.g.   citing   studies   or   historical   examples   that   the   human

opponent had not remembered)

45

. In the post-debate poll, many audience members said the AI enriched

their knowledge, though the human had better delivery and charisma. In an experimental evaluation across

20+ unseen topics (reported in the 2021 Nature paper), judges rated Debater as generating  high-quality

arguments  and sometimes winning on  knowledge and substance, but humans won on  persuasiveness and

delivery.   IBM   noted   that   in   a   competitive   debate   setting,   Project   Debater   achieved   something
unprecedented:  full  automation  of  a  task  involving   natural   language   reasoning.   It  did   manage   to  beat

some   human   debaters   in   testing   scenarios,   especially   on   informing   the   audience

46

,   but   it   did   not

consistently outperform top experts in rhetorical skill. The key metric was not a simple objective score but a

qualitative assessment of argument quality and audience reception. By those measures, Project Debater

was at human level in content richness, but a notch below in humor, emotional connection, and adaptive

rebuttal finesse. Notably, it did not devolve into incoherence; it maintained logically structured arguments, a

significant AI feat in unrestricted language.

Q4   Failure   Modes:  Project   Debater’s   limitations   became   evident   in   its   bouts   with   humans.   It   lacked

emotional intelligence and humor – components often crucial in persuasion. The human champion used

spontaneous wit and relatable analogies, whereas Debater’s style was formal and info-dense, which can

alienate   listeners.   The   AI   also   sometimes   produced   arguments   that   were  logically   relevant   but   not

strategically   optimal.   For   example,   it   might   bring   up   a   minor   supporting   fact   while   missing   a   major

intuitive   point   that   sways   humans.   This   highlights   AI’s   lack   of   a   true   understanding   of   what   convinces

people. Debater also occasionally made  errors in comprehension  – misinterpreting a point the human

made, or its rebuttal would miss the crux of the opponent’s argument if that argument relied on subtle

implications or humor. Additionally, because it was corpus-based, if a topic had sparse or biased coverage in

its database, Debater might stumble or reflect those biases. An example: in one debate it argued for a topic

using mainly dry statistics, missing more persuasive anecdotal appeals a human might use. And unlike a

human, Debater couldn’t dynamically change its style mid-debate – it had a fixed, somewhat monotonic

speaking tone and pace, which impacted its convincingness. In summary,  humans still prevailed in the

“art”   of   debate,   leveraging   creativity,   emotional   appeal,   and   adaptability,   whereas   Debater   sometimes

came off as a well-informed but one-dimensional speaker

47

.

Q5   Transferability:  The   technologies   behind   Project   Debater   have   clear   applications.   Elements   have

already spun off into products: the ability to digest large volumes of text and pull out pro/con arguments is

useful for decision support (e.g. an AI tool that briefs a CEO on all arguments for and against a policy).

Debater’s speech composition could assist in generating reports, executive summaries, or even drafting

persuasive   essays   –   essentially   automating   parts   of   white-collar   work   that   involve   writing   and

argumentation. In negotiation settings, an AI could use Debater-like capabilities to prepare negotiation

positions or real-time rebuttals to an opponent’s points (though understanding tone and intent is still a

12

challenge). However, a fully autonomous debating agent is not directly transferable to real negotiations or

court arguments yet; those involve back-and-forth interaction with unpredictable human input, something

Debater   could   only   handle   in   a   limited   way.   The   project   did   illustrate   how   an   AI   might   participate   in

meetings or strategy discussions  as a factual knowledge source that argues a certain viewpoint. In the

long run, if combined with improved emotional modeling (maybe via advanced language models), such

systems could become persuasive communicators – a key aspect of a “billionaire AI” would be selling its

ideas to humans, after all. For now, Debater’s legacy is as a proof-of-concept that AI can engage in high-

level  cognitive  discourse,  even  if  humans  maintain  an  edge  in  creative  and  emotional  appeal.  IBM’s  team

themselves noted that debating lies “in a different territory, in which humans still prevail, and novel paradigms

are required to make substantial progress.”

47

.

Q6 Citation:  Slonim et al. (IBM) published  “An autonomous debating system”  in Nature 2021. The abstract

concludes: “Classical ‘grand challenges’ like games lie in the comfort zone of AI, whereas debating with humans

lies in a different territory in which humans still prevail, requiring new paradigms for progress.”

47

.

2022: Meta CICERO – Diplomacy and Negotiation with Humans

Q1 Domain & Analogy: CICERO, developed by Facebook (Meta) AI with academic collaborators, was the first

AI to achieve human-level performance in the board game  Diplomacy. Diplomacy is a 7-player strategy

game about early 20th-century geopolitics; it famously requires players to negotiate, form alliances, bluff,

and   occasionally   betray  to   conquer   territories.   Unlike   purely   competitive   games,   Diplomacy   involves

natural language communication among players – making it an analog for international diplomacy (as the

name suggests) or complex  multilateral negotiations  in business and politics. A successful Diplomacy

agent must combine strategic planning (like in chess or Go) with negotiation skills akin to real-world deal-

making. Achieving this means an AI can cooperate and compete simultaneously, navigating alliances and

trust dynamics, much as a business negotiator might in mergers, cartel formations, or any scenario with

multiple parties and persuasion.

Q2 Design & Architecture: CICERO’s architecture fused two main components: a strategic reasoning engine

and a natural language generation (NLG) engine. The strategic module was built as a planning agent (similar

in spirit to game-playing AIs) that could predict others’ actions and decide optimal moves for the board

game given the current state and any commitments made. The language module was based on a fine-

tuned large language model (LLM) which could generate dialogue – messages to send to other players –

and interpret their messages. During each turn of Diplomacy, CICERO would take in the state of the board

and the transcript of messages so far, then come up with an intended plan. It would then craft messages to

different players to negotiate, such as proposing alliances or coordinating moves. The LLM was trained on

an annotated corpus of Diplomacy games including human dialogue (Meta collected games on an online

platform with humans who chatted in a controlled setting). It had to be contextually aware: for instance, if it

promised Player A to attack Player B, it should not contradict itself later. The architecture used  dialogue

coherence checks to ensure its messages were consistent with its actual plans. It also maintained models

of each human player’s likely actions (a form of opponent modeling) to strategize effectively. Technically,

CICERO was a marriage of multi-agent RL (to handle moves) and NLP (to handle language), with iterative

processing between the two. This combination allowed CICERO to decide not just what to do, but what to say

to others – a significant step towards autonomous AI agents that communicate and act.

Q3 Performance:  Between August and October 2022, CICERO was tested in 40 online Diplomacy games

with human players on webDiplomacy.net (an online platform where anonymous players play Diplomacy via

13

text chat). The results were striking: CICERO achieved a rank in the top 10% of players who had played at

least one game in that period

48

. In terms of game outcomes, CICERO won some games outright and

more frequently finished with strong draws, accumulating a high average score that equated to elite human

performance.   Importantly,   the   human   players   did   not   know   which   participant   was   the   AI,   and   many

presumed CICERO was human – it was able to “trick humans into thinking it was real” by engaging in natural

dialogue and making credible plans

49

. Observers noted CICERO would invite players to alliances, suggest

coordinated   moves,   and   negotiate   peace   deals   convincingly

49

.   One   example:   CICERO   might   message

country X, “Let’s form a truce; I move out of region Y so you can trust me, and together we’ll attack country

Z next turn.” It often followed through, at least until it no longer served its interest, much like a human

would. The AI’s strategic play was solid as well – by combining communication with tactical calculations, it

could   outsmart   opponents.   An   article   in  Science  reported   that   CICERO   sometimes   even   out-negotiated

humans, leveraging its perfect recall of past promises and ability to do planning of several moves ahead in

negotiations

50

51

.   Its   biggest   achievement   was   demonstrating   coherent,

 truthful-yet-strategic

dialogue: Meta found that blatant deception backfired for the AI, so it learned to mostly keep its word or at

least not get caught lying (when CICERO was too deceptive, its gameplay success dropped, so they adjusted

. This aligns with human diplomatic wisdom: credibility is a currency. In short,
it to be more honest)
CICERO operated at a human expert level in a complex game requiring both sharp strategy and linguistic

52

negotiation – a first of its kind.

Q4 Failure Modes:  While CICERO was highly competent, it was not invincible. Skilled humans could still

sometimes   detect   that   something   was   off   –   for   instance,   CICERO’s   language,   though   fluent,   could

occasionally seem a bit formulaic or overly agreeable. If players suspected an ally might be an AI, they could

test it in ways an AI might struggle with (like making an odd joke or referencing out-of-game knowledge).

CICERO also had the limitation of not truly “understanding” the semantics of the language beyond pattern

prediction. In one internal test, it apparently made an offer that was technically logical but violated common

sense expectations, confusing a human. When it did decide to betray an ally (which is often necessary to

win in Diplomacy), it had to word its betrayal carefully; if not, it risked uniting others against it. Additionally,

CICERO’s strategy, while strong, is tied to the specifics of Diplomacy rules – a human might adapt to a novel

rule or negotiate creative arrangements outside the game’s standard bounds, whereas CICERO wouldn’t

know   how   to   handle   anything   off-script.   There   were   also   ethical   failure   considerations:   an   AI   that   can

manipulate   humans   raises   flags   (Meta   emphasized   it   did   not   allow   toxic   language   and   tried   to   keep

CICERO’s persuasion within fair bounds). Technically, one failure mode occurred when there was too much

for the AI to communicate; with seven players, if dialogue volume got high, CICERO sometimes struggled to

respond promptly to everything, potentially missing critical info. In general, though, no glaring exploit was

publicized – the human-level nature of CICERO meant it had no obvious, easy weaknesses beyond what a

human leader in the game might have (like trusting the wrong person or mis-estimating a threat).

Q5 Transferability: CICERO represents a leap toward AI in complex negotiation and cooperation settings.

Its architecture of combining strategic planning with natural language interaction is directly applicable to

many domains: international diplomacy (as the game itself mirrors), business negotiations (AI agents could

negotiate deals, contracts, or resolve disputes via dialogue), and task delegation (an AI coordinating among

groups of humans by negotiating tasks and responsibilities). A potential early application is AI support in

online negotiations or bargaining platforms – CICERO-like agents might mediate or suggest resolutions.

Another area is virtual assistants that can interact with multiple users or other AI to coordinate schedules or

agreements (like scheduling meetings between many busy people by negotiating times). The transfer to

real-world use, however, faces challenges: real negotiations often involve complex utilities (emotions, long-

term relationships, legal constraints) that were absent in the self-contained game of Diplomacy. Also, open-

14

ended human language in negotiations can be far more nuanced (sarcasm, implicit threats, etc.) than the

relatively goal-focused Diplomacy chats. Nonetheless, CICERO’s success strongly suggests that  language-

capable   strategic   AIs  can   be   built.   This   is   a   foundational   capability   for   a   “billionaire   AI”   –   running   a

company involves not just optimizing decisions but communicating with partners, clients, regulators, and

employees.   An   AI   CEO   would   need   a   Cicero-like   ability   to   persuade   and   collaborate.   On   the   flip   side,

CICERO’s existence also flags possible malicious uses (e.g. automated scam negotiations, as some experts

noted

53

).   Overall,   the   techniques   (LLM   for   dialogue   +   planning   algorithm   for   strategy)   will   likely   be

extended to other multi-agent interactions. We might soon see AI negotiators in limited domains like supply

chain contracts or ad auctions, where they converse and bargain in real-time to settle on outcomes that

humans currently negotiate.

Q6 Citation: Meta’s CICERO was reported in Science in November 2022. The Washington Post wrote: “Meta’s

AI agent CICERO… was able to trick humans into thinking it was human, inviting players to alliances, crafting

invasion plans and negotiating peace deals when needed. To test it, Meta let Cicero play 40 games… it placed in

the top 10 percent of players, the study showed.”

49

48

2022: Salesforce AI Economist (v2) – Optimizing Tax Policy via

Simulation

Q1 Domain & Analogy: The AI Economist is an economic policy design simulation developed by Salesforce

Research, with the second version (v2) reported in 2022–2023. The domain is a simplified model of a small

economy – multiple agents (which can represent citizens) earn income via labor and a government sets tax

rates   to   redistribute   income.   The   goal   often   is   to   optimize   for   some   social   welfare   function,   balancing

equality   and   productivity.   This   is   directly   analogous   to  real-world   tax   policy   and   economic   planning:

governments try to set tax rates to maximize societal objectives (e.g. minimize inequality without killing

incentives). The AI Economist essentially creates a virtual economic-policy lab where AI can experiment with

tax schemes. In broader business terms, it touches on algorithmic governance and mechanism design –

akin to an AI autonomously setting policies in a market, a potential future role for AI in public policy or

corporate pricing strategies.

Q2 Design & Architecture: The AI Economist v2 uses a two-level deep reinforcement learning framework

3

.   At   the   lower   level,   you   have   multiple  learning   agents  (simulated   “people”)   who   make   economic

decisions – e.g. work vs leisure, trading, moving in a grid-world economy – to maximize their personal

utility. These agents can be standard RL agents trained to respond rationally to taxes and opportunities. At

the higher level, there is a  social planner agent  (the AI economist) that adjusts the tax policy (tax rates on

income brackets, possibly subsidies) in response to the agents’ behaviors, with the aim of maximizing a

social   welfare   objective   set   by   the   researchers   (like   a   weighted   sum   of   equality   and   productivity).   The

planner observes state statistics (like income distribution) and periodically proposes new tax rates, then

sees outcomes. Both levels are trained with reinforcement learning in an iterative loop: the social planner

learns how to set taxes anticipating agent reactions, and agents learn how to work and earn given the tax

policy. It’s essentially a  bi-level optimization: the government AI and the populace co-evolve. In version 2,

improvements included more realistic settings (e.g. spatial layout for agents to move and trade) and using

techniques like curriculum learning to stabilize training. The architecture was tested via simulation episodes

representing “years” of an economy. No actual humans are in the loop; this is a fully simulated environment,

but the novelty is treating policy design as a learning problem. (In one sense, the AI Economist can be seen

as playing a game against an abstract adversary: inequality and inefficiency).

15

Q3 Performance: The AI Economist was able to discover tax policies that outperformed several baselines,

including   standard   economic   formulas   and   human-designed   tax   schemes

3

.   In   their   2022   Science

Advances  paper,  the  researchers  report  that  the  AI-designed  tax  schedule  achieved  a  better  balance  of

equality   and   productivity   –   specifically,   they   often   used   a   social   welfare   measure   where   0   means   no

redistribution and 1 is full equality, and the AI could reach higher values than both a current US Federal

income   tax   schedule   and   an   optimized   linear   tax   formula.   For   example,   in   one   scenario   the   AI   policy

improved social welfare by ~16% over the best-known alternative policy

54

. It did so by implementing a

dynamic tax that sometimes is counter-intuitive (e.g. slightly regressive segments that ultimately incentivize

more productivity and thus more total pie to redistribute). In short, AI achieved a more efficient equality-

productivity   trade-off  than   humans   had   via   simple   rules.   This   suggests   AI   found   a   sweet   spot   where

people   still   want   to   work   but   income   disparity   is   moderated   better.   The   AI   policies   also   tended   to   be

dynamic: adjusting tax rates based on economic conditions, something that’s hard for static human policies.

An earlier version (v1 in 2020) already showed improvement, and v2 solidified this, adding realism and

showing robustness. Salesforce highlighted that “AI does it better, when it comes to optimal tax policy design”,

though with the caveat that the AI optimizes the given welfare objective

54

. It’s noteworthy that when

tested with human participants (in a limited experiment outside of v2, involving humans in the loop), the AI
could adapt the taxes to human behavior in simulation and still improve outcomes – indicating potential

generalization.

Q4 Failure Modes:  One fundamental limitation is the  simplification  of the simulation. The AI Economist’s

world is a toy compared to actual economies: agents have very limited behavior (work or not, move on grid,

trade basic resources), there’s no rich innovation or complex capital. As a result, policies that work there

might   not   translate   directly   to   real   society.   A   failure   mode   in   simulation   was   that   if   agents   learned

pathological behaviors, the planner might overfit to them; the team had to ensure variety in agent types to

keep the policy robust. There’s also the question of multiple objectives – if the objective were different (say

prioritizing growth more), the AI would produce a different policy. So it’s only as “good” as its objective

function;   mis-specified   objectives   could   yield   undesired   policies   (like   extremely   high   taxes   and   then

subsidies   that   achieve   equality   but   might   be   politically   unacceptable   or   unfair   in   unmodeled   ways).

Additionally, the AI’s policy, while optimal in simulation, might be hard to explain – it could set up a complex

piecewise tax scheme that a human policymaker would struggle to interpret or justify to the public. This

lack of interpretability and potential unfairness (if certain agents by luck benefit or suffer) is a risk. In v2,

they also found that if agent behavior deviates from assumptions (e.g. humans don’t work exactly like RL

agents), the policies might not be as effective – a sim-to-real gap. A practical failure mode: political feasibility

– the AI might suggest something like 55% tax on mid-income and 0% on very low and very high, which

could be optimal in sim but would face resistance (the AI of course doesn’t account for political backlash or

legal constraints). Technically, a failure is that training the two-level system can be unstable (the planner and

agents are co-learning, which is like a game equilibrium problem); ensuring convergence required careful

tuning.

Q5  Transferability:  The  AI  Economist  framework  is  very  transferable  in  concept:  any  situation  where  a

centralized   policy   interacts   with   multiple   learning   agents   can   use   a   similar   two-level   RL   approach.   This

includes mechanism design problems like setting tolls in traffic systems (agents = drivers), auction designs

for   ad   markets   (agents   =   bidders,   policy   =   pricing   rule),   or   setting   rules   in   a   trading   market   (agents   =

traders,   policy   =   transaction   tax   or   regulation).   In   operations   management,   one   could   envision   an   “AI

Manager” adjusting workflow rules while worker agents (human or AI) respond – analogous to AI Economist

adjusting taxes while workers respond. The framework of having AI “social planner” could also apply in a

firm   for   resource   allocation:   imagine   an   AI   dynamically   setting   budgets   or   bonuses   for   divisions   to

16

maximize the company’s overall output. These are similar mathematically to a tax problem. Additionally, the

AI Economist hints at a future where AI might assist governments in policy: it can sift through vast scenario

simulations to propose policies optimized for stated goals, serving as a decision-support tool (as noted, the

intent is to augment human policymakers

54

, not fully replace them, because humans must set the goals

and constraints). Another transferable insight is how the AI can handle the unpredictability of adaptive

agents – essentially modeling human behavior in response to policy. This could be extended to any context

where policy affects human behavior (climate policy affecting industry, monetary policy affecting banks,

etc.). The success of AI Economist demonstrates that AI can autonomously discover creative policy solutions

– a step toward AI-driven governance or economics. One enabler is the increasing availability of computing

to run such simulations. However, transferring these results to the messy real world remains a big leap,

since real economies have chaos and complexity not captured in gridworld simulations. Still, the trend is

that as simulations improve (or AIs learn from real data directly), an AI could potentially manage aspects of

an economy or a company’s strategy in a data-driven optimal way that humans, with their simpler rules of

thumb, cannot.

Q6 Citation: Zheng et al. (2022) in Science Advances presented these results. A Salesforce blog TL;DR noted:
“The AI Economist, a RL system, learns dynamic tax policies that optimize equality along with productivity in

simulated economies, outperforming alternative tax systems.”

3

. In other words, given a social objective, the

AI   found   better   tax   plans   than   humans   have   –   a   striking   instance   of   AI  autonomously   governing   an

economy (albeit a virtual one).

2024: Cambridge Judge Experiment – Generative AI as an

Automotive CEO

Q1 Domain & Analogy: In early 2024, researchers at Cambridge Judge Business School (with HBR) tested AI

in a strategic business simulation of the automotive industry. The simulation was a gamified replica of a

competitive market, presumably the U.S. auto sector, including real historical data on pricing, market shifts,

sales, and macroeconomic trends like the COVID-19 shock

55

. Human participants, including executives

and  MBA  students,  played  as  CEOs  making  decisions  about  product  design,  pricing,  marketing,  etc.,  to

maximize their firm’s market share and profitability. Crucially, the experiment introduced a Generative AI

agent (based on GPT-4) as one of the “CEO” decision-makers. This domain directly analogizes to running a

company in the real world. The question: Can AI make better strategic decisions than human CEOs under

the same conditions? This scenario is basically a controlled trial of AI-led corporate management.

Q2 Design & Architecture: The AI CEO was implemented using OpenAI’s GPT-4 (referred to as “GPT-4o” in

some summaries

56

, possibly GPT-4 with certain optimizations). The AI was connected to the simulation

environment (nicknamed “The Electric Car Revolution” digital twin

56

). Each decision round, the AI likely

received a prompt describing the state of the company and market (sales numbers, competitor moves,

economic   indicators)   and   was   asked   to   output   strategic   decisions   (e.g.   set   price   for   Model   X,   allocate

marketing budget, choose whether to invest in EV technology, etc.). The Strategize Inc team (who built the

simulation) crafted a  “Strategy Meta Grid”  or “Dojo” that allows LLMs to compete in this environment

57

.

GPT-4   was   fine-tuned   or   instructed   to   pursue   profit   and   share,   making   data-driven   choices.   Humans

similarly made decisions through a game interface. The AI could also be set to run multiple companies (or

an ensemble of different strategic styles) to test various approaches. While details of architecture aren’t fully

public, it’s clear  generative AI was used for its reasoning ability on business data  – GPT-4 can ingest

large data about historical trends and presumably weigh options (somewhat like a very advanced decision

17

support  tool,  but  acting  autonomously).  The  design  ensured  the  AI  had  the  same  info  as  humans  and

operated under the same timing and rules. They also included “black swan” events (like a sudden economic

downturn or supply chain disruption) to see how AI vs humans react

58

59

.

Q3 Performance:  The experiment found that the  AI models outperformed human participants  on key

metrics   of   strategic   decision-making   in   the   simulation

60

4

.   Specifically,   the   AI   CEO   achieved   higher

profitability and market share on average compared to the human CEOs

60

. It excelled at data-driven tasks

– analyzing market trends, optimizing prices, choosing product mixes – leveraging its capacity to absorb

and compute information. For instance, generative AI could identify subtle demand patterns or factor in

economic trends (from its training knowledge) that humans overlooked. One result was that AI improved

the strategic planning process and avoided certain costly mistakes humans made

61

. However, the AI

was not perfect: notably, it faltered in dealing with unpredictable disruptions (“black swan” events)

62

.

When   an   unforeseen   scenario   (maybe   a   pandemic-like   sales   crash   or   supply   shock)   occurred,   the   AI’s

performance dipped more relative to humans, who could leverage intuition or creative thinking. The overall

takeaway   reported   was:  “Generative   AI   can   significantly   outperform   human   CEOs   in   data-driven   tasks…and

.
prevent mistakes, but cannot yet assume the full role of a CEO, especially in markets that serve humans.”
Humans still held an edge in soft areas like empathy, ethical judgment, and handling ambiguity, which are

61

crucial   in   real   leadership

63

.   The   AI’s   impressive   showing   nonetheless   suggests   that   for   many   routine

executive decisions (pricing, investment allocation given data), AI can do as well or better. In terms of hard

numbers: the HBR article title says “AI Can (Mostly) Outperform Human CEOs” – implying mostly metrics

were better for AI, possibly by a significant margin, though we don’t have exact percentages. There were

344 participants  in the months-long simulation, indicating a robust sample

55

. The AI’s consistency and

evidence-based   strategy   were   praised   as   it   “significantly   improved   profitability   and   market   share”   on

average

4

.

Q4   Failure   Modes:  The   key   failure   mode   observed   was   handling   the  unpredictable   and   the   human

elements. When a black swan event hit (e.g., a sudden regulatory change or a new tech rendering current

strategy obsolete), the AI lacked an internal model of such novelty – it might have continued optimizing

based on outdated assumptions, whereas some humans exercised creativity or prudent caution. Also, the

AI grew overconfident in stable scenarios

57

64

. According to the LinkedIn commentary by the authors,

AI outperformed in nearly every metric but “could not beat humans on long term sustainability as it grew

overconfident”

57

.   Overconfidence   could   mean   it   doubled   down   on   a   strategy   that   worked   historically

without   hedging   against   potential   future   shifts   –   something   an   experienced   CEO   might   be   wary   of.

Additionally,  AI  lacks  empathy  and  ethical  judgment:  the  HBR  piece  noted  AI  cannot  replace  CEOs  in

serving human markets because leadership involves motivating people, brand vision, and ethics

63

. In the

sim, these factors might be abstracted away, but in reality an AI might make a decision that’s profit-optimal

but causes public backlash or hurts employee morale, which a human might foresee. Another limitation was

contextual understanding: if the simulation threw in scenarios requiring common sense outside of data

(like  a  labor  strike  or  geopolitical  issue),  the  AI  might  not  respond  appropriately  unless  such  scenarios

appeared in its training data. The experimenters concluded that while AI is great with data-rich analysis, it

“falters   in   unpredictable   disruptions”

62

,   underscoring   that   it   doesn’t   truly   comprehend   events   –   it

predicts based on patterns. Finally, one must consider that the AI had no accountability or fear – a human

CEO might take a cautious strategy to avoid catastrophic risk (even at cost of some profit), whereas the AI

might take an aggressive bet to maximize metric in sim. In real life, those bets could backfire disastrously

(and a human would have to bear consequences, an AI can’t).

18

Q5 Transferability: This study is perhaps the most direct evidence to date of AI’s role in corporate strategy

and operations. The results suggest in the near term, AI can be used as a powerful decision-support tool:

e.g. a human executive teaming up with an “AI planner” to evaluate decisions. Over time, as trust grows, AI

might autonomously manage more business functions. Transferability is high in any  data-rich strategic

domain: such as inventory and supply chain planning, financial portfolio management (similar decisions

but in finance), marketing spend optimization, etc. Indeed, the authors forecast  “The rise of ‘artificial CEOs’

could   disrupt   traditional   strategy   consulting…   firms   like   McKinsey   may   find   services   supplemented   or   even

replaced   by   AI   systems   tailored   to   client   ecosystems.”

65

.   That   quote   suggests   AI   advisors   might   handle

analysis that consultants do. Moreover, the experiment’s “Strategy Dojo” platform can be adapted to other

industries – one could simulate telecom, pharmaceuticals, etc., and test AI vs human managers. If similar

outcomes occur (AI besting humans in analytic decisions), companies might adopt AI for scenario planning

and  strategy  generation.  However,  full  transfer  to  a  billionaire-level  AI  agent  running  a  real  company

entails   more   than   simulation:   it   requires   real-world   sensing,   adaptability   to   non-modeled   events,   and

human buy-in. Legally, an AI can’t be a CEO (yet), and stakeholders wouldn’t accept major decisions with

ethical ramifications being made by a machine without oversight. So in practice, the near-term transfer is

“AI as co-pilot” for executives. Enablers include that modern enterprises have lots of data and operate partly
in  digital  realms  (e.g.  e-commerce,  logistics),  which  AI  can  leverage.  If  a  company  gave  an  AI  real-time

dashboards and authority to adjust certain levers (prices, supply orders), you could get an autonomous

division manager AI fairly soon. The barrier remains that humans must handle areas AI is weak at (inspiring

employees, adapting to society). In summary, the Cambridge experiment demonstrates  feasibility of AI-

driven strategy in complex, realistic business settings, marking a concrete step towards autonomous

business  agents.  The  recommended  model  is  hybrid:  “AI  complements  human  CEOs,  who  focus  on  vision,

values, and long-term sustainability”

66

  – implying transfer will happen in a collaborative form, not pure

replacement, at least in the medium term.

Q6 Citation:  Mudassir, Munir, Ansari & Zahra summarized in  Harvard Business Review  (Sept 2024) that  “AI

models bettered human participants in strategic decision-making involving profitability and market share, but

falter   in   dealing   with   unpredictable   disruptions…despite   impressive   performance,   AI   cannot   assume   full   CEO

responsibility in markets that serve humans.”

67

68

.

2025: Andon Labs Vending-Bench – Stress-Testing Long-Term

Autonomy

Q1 Domain & Analogy: Vending-Bench  is a 2025 benchmark introduced by Andon Labs to evaluate an AI

agent’s ability to  run a simple business over an extended period. The scenario is operating a  vending

machine   business:   the   agent   must   manage   inventory   (order   products   from   suppliers),   set   prices,   and

ensure profitability while paying daily fees – essentially a mini entrepreneurial task

69

. This domain stands

in   for   the   broader   challenge   of  long-horizon   decision-making   in   operations.   A   vending   business   is

conceptually straightforward (each task is simple), but doing it well over many cycles requires consistency

and memory – akin to running any small business or a portion of a supply chain continuously. The key

analogy is to “long-term coherent autonomy” in business: can an AI not just make one good decision, but

string together thousands of good decisions without losing track of its goal?

Q2 Design & Architecture:  Vending-Bench’s novelty is in how it tests AI: it uses  Large Language Model

(LLM)-based   agents  (like   GPT-4   variants,   Claude   from   Anthropic,   etc.)   as   the   brains   operating   within   a

simulated text-based environment. The agent interacts with the sim through text (e.g. it might receive a

19

daily report: “You have X stock, prices are Y, a supply delivery is due, etc.” and then it outputs actions: “Order

10 sodas, set soda price $1.50.”). The benchmark runs for extremely long dialogues – the paper mentions

>20 million tokens per run

5

, which is orders of magnitude longer than typical LLM context windows. The

focus is on an architecture that may include an LLM augmented with memory mechanisms (like keeping

notes or using external tools) to stay coherent over time. Andon Labs tested various models (Claude 3.5,

GPT-4 derivatives, and their own “o3-mini”) by letting them act as the vending operator. They also had a

human baseline  – presumably a human performing the same simulation for comparison

70

71

. Agent

design might involve prompting the LLM in a loop, possibly with a chain-of-thought. Given the nature of the

task, this isn’t about fancy neural networks beyond the LLM; it’s more about whether the LLM can remain

goal-focused   and   avoid   drift.   The   environment   simulates   day-by-day   events,   including   supplier

communications (perhaps an email like “shipment will arrive in 3 days”) and customer purchases. The agent

must   remember   these   future   events   and   not   “forget”   to   stock   or   collect   revenue.   Scoring   is   based   on

profitability achieved and whether the agent avoids bankrupting the business.

Q3 Performance:  The findings were that current LLM agents show  high variance  in performance

72

. In

many   runs,   especially   the   best   ones,   top   models   like   Anthropic’s   Claude   3.5   (“Sonnet”   version)   and   an
optimized GPT-4-based model (“o3-mini”) were actually able to operate the vending machine effectively

and turn a profit, even outperforming the human baseline in some runs

71

. This demonstrates that AI

can indeed learn the pattern – for example, ordering stock just in time, setting profitable prices – and

sometimes   even   find   better   policies   (perhaps   adjusting   prices   more   dynamically   than   the   human   did).

However, all models had failure runs where things went off the rails

72

. Common failure modes included:

misinterpreting   the   delivery   schedule   (e.g.   forgetting   that   an   order   was   arriving   and   ordering   again

unnecessarily), forgetting to reorder stock (leading to empty machine and lost sales), or falling into bizarre

“meltdown loops”  of incoherent behavior

73

. A meltdown loop might be the LLM going on a tangent

unrelated to the task – since these models can drift if not carefully grounded. Once the agent derails, it

rarely   recovers   without   external   reset

73

.   Interestingly,   the   researchers   found  no   clear   correlation

between these failures and the context window limits  being hit

74

. In other words, the breakdowns

didn’t always happen just because the LLM’s memory filled up; they sometimes happened much earlier,

indicating an inherent challenge in long-term focus. On average, the human baseline was more consistent

(humans didn’t completely forget the business objective), whereas AI had a wider performance spread –

some runs great, some disastrous. The  best AI runs did exceed human performance  in profit, showing

the   potential   upside   of   tireless,   fast   computation

71

,   but   the   reliability   issue   dragged   down   average

performance. Another result: Vending-Bench explicitly tested whether giving the agent more memory (like

letting it periodically summarize or use scratchpad notes) helped; those details are technical, but the core

remains that current state-of-the-art LLMs struggle to maintain long-term coherence in decision-making.

Q4 Failure Modes: As noted, the prime failure modes were forgetting and misinterpreting over long horizons.

An agent might confuse inventory numbers as the days go by or lose track of a debt that’s accumulating.

Others   would   start  generating   off-topic   content  –   e.g.,   instead   of   focusing   on   vending,   the   AI   might

ramble   about   unrelated   matters   (a   known   LLM   issue,   where   given   enough   turns   it   might   treat   the

simulation   like   a   conversation   and   drift   into   storytelling).   Some   agents   entered   repetitive   loops   of

apologizing   for   mistakes   or   re-hashing   plans   without   executing   them   (like   analysis   paralysis).   These

“meltdowns”  are  qualitatively  similar  to  cognitive  fatigue  or  distraction  in  humans  –  except  much  more

pronounced for AIs after prolonged operation. Another failure mode was  misaligned incentives: if not

carefully instructed, an AI might try odd strategies, like ordering an excessive amount of product due to

some misunderstanding of scoring (some models seemed to not fully grasp the need to make profit and

instead just avoid stockouts by overstocking massively, leading to costs that bankrupt the operation). The

20

study also hints that the failures did  not come primarily from running out of context memory

74

 – so it’s

not just that the prompt got too long; it’s an inherent inability to maintain strategy over long periods. This

suggests something like an “attention drift” problem in LLMs. Another observed issue: some runs the AI

mis-handled the timing – e.g., if a supplier delivers every Monday but the AI forgot and double-ordered on

Tuesday after panicking about low stock, that run would spiral. Humans rarely made that exact error. These

kinds of lapses illustrate that robust long-term planning under even mild uncertainty is still a challenge

for AI.

Q5   Transferability:  Vending-Bench   was   explicitly   created   to   test   capabilities   needed   for   “dangerous

hypothetical scenarios” – in particular the ability of an AI to  acquire and manage capital autonomously

75

76

. This is a core skill for any would-be “billionaire AI.” If an AI can reliably run a profitable small

business for months on end, one could scale that up conceptually. The benchmark highlights what’s missing

for   that:   more   stable   long-term   reasoning.   The   insights   from   Vending-Bench   likely   transfer   to   any

environment where an AI must operate continually without resets – such as an AI manager running 24/7, or

an AI agent that continues to execute tasks beyond a single session (like AutoGPT trying to continuously

self-improve).   The   findings   push   AI   developers   to   incorporate   better   memory   architectures   or   periodic
reflection phases to avoid drift. In a sense, Vending-Bench is a microcosm of running any enterprise: lots of

routine,   occasionally   an   important   event,   and   the   necessity   not   to   drop   the   ball.   Transfer-wise,

improvements  that  let  an  AI  sustain  coherent  goals  over  millions  of  tokens  would  benefit  all  domains

requiring endurance – from continuous process control (like running a power plant) to long dialogues (like

personal   AI   assistants   that   remember   a   user’s   life   details).   Also,   the   benchmark   underlines   that   simply

scaling   model   size   or   context   might   not   solve   this;   it   may   need   new   approaches   (e.g.,   splitting   tasks,

external long-term memory, etc.). On the positive side, the fact some runs did well indicates AIs can do long-

term tasks under ideal conditions, so with further refinement they might become far more reliable. For AI

autonomy, solving this is critical – an autonomous business agent that goes off the rails even 5% of the time

could cause huge real-world damage. Thus, Vending-Bench serves as a  yardstick for progress toward

safe, effective long-duration autonomy. As AI researchers address the failure modes (perhaps by giving

the agent better self-monitoring or the ability to reset its focus), we can expect transfer to more complex

business tasks.

Q6   Citation:  Backlund   &   Petersson   (Andon   Labs)   released  “Vending-Bench:   A   Benchmark   for   Long-Term

Coherence of Autonomous Agents”  (Feb 2025). They write:  “Agents must balance inventories, place orders, set

prices, and handle daily fees – tasks that are each simple but collectively, over long horizons (>20M tokens), stress

an LLM’s capacity for sustained, coherent decision-making. Our experiments reveal high variance: Claude 3.5 and

others   manage   the   machine   well   in   most   runs   and   turn   a   profit,   but   all   models   have   runs   that  derail  –

misinterpreting delivery schedules, forgetting orders, or descending into tangential ‘meltdown’ loops… Some runs

with the most capable LLMs outperform the human baseline, albeit with higher variance.”

69

71

.

Cross-Study Synthesis by Domain

AI  vs  human  performance  in  these  studies  shows  clear  patterns  across  different  business  and  strategy

domains:

21

Operations & Long-Term Management

In operational tasks requiring sustained attention to detail (inventory management, scheduling, repetitive

decision cycles), AI can execute diligently and often optimally – outperforming humans in consistency and
data processing – but struggles with long-term coherence. For example, in the Vending-Bench operations

simulation, AI agents sometimes surpassed human profit by optimizing pricing and stock decisions, proving

superior in routine optimization

71

. However, they also exhibited  high variance  and eventual breakdowns

that a human operator would not

72

. Humans excel at maintaining steady performance and common-

sense consistency over long horizons; current AIs are prone to compounding errors or losing focus after

many steps. Similarly, the Cambridge auto industry experiment showed AI (GPT-4) excelling in data-driven

planning   –   preventing   short-term   mistakes   and   improving   efficiency   –   yet   it   faltered   when   unexpected

disruptions required a re-think or a gut call

67

59

. In summary, AI outperforms in stable, information-

rich operational settings (e.g. precise supply chain tweaks, dynamic pricing), but human managers

outperform in  volatile or prolonged scenarios  that demand adaptability, institutional memory, and tacit

knowledge.   Going   forward,   hybrid   approaches   are   emerging:   let   AI   handle   granular   decisions   and

monitoring, while humans set high-level goals and intervene at signs of divergence or when novel events

occur

66

.   This   complements   AI’s   tirelessness   with   human   judgement.   Achieving   reliable   long-term

autonomy   will   require   technical   advances   (e.g.   memory   architectures   to   give   AI   better   “cognitive

endurance”) to close that gap.

Finance & Economic Policy

In domains of finance and economics, AI has shown a strong ability to optimize quantitative objectives and

discover strategies that improve on human heuristics. The Salesforce AI Economist is a prime example: it

autonomously learned tax policies that yielded a better equality-vs-productivity tradeoff than both status

quo policies and known economic formulas

3

54

. This indicates that for tasks like  tax policy design,

resource allocation, and potentially investment portfolio optimization, AI can identify solutions in high-

dimensional spaces that humans overlook. Notably, AI Economist’s policy outperformance was on the order

of ~16% improvement in social welfare – a non-trivial gain in economics

54

. In finance, while not detailed

above,   there   is   a   long   trend   of   algorithmic   trading   systems   outperforming   human   traders   in   certain

contexts  (high-frequency  trading,  quantitative  strategies).  However,  humans  remain  dominant  in  setting

goals   and   handling   qualitative   factors   in   economic   policy   (e.g.   political   feasibility,   fairness   beyond   the

model’s   scope).   AI’s   advantage   lies   in  complex   optimization   under   clear   objectives:   it   will   reliably

maximize   defined   metrics   (return,   welfare,   risk-adjusted   profit),   often   avoiding   the   behavioral   biases

humans   have.   On   the   other   hand,   humans   are   better   at  incorporating   unmodeled   factors  –   e.g.,   a

government might reject an “optimal” tax policy because of public sentiment or moral values, which an AI

would not consider unless explicitly encoded. Another point is interpretability: AI solutions in finance/policy

(like a weird tax curve or trading strategy) might be hard to justify to stakeholders. So humans currently

retain   trust   and   final   authority.   Nonetheless,   we   see   AI   increasingly   used   to   assist   human   decisions:

governments are exploring AI-driven simulations for policy analysis, and banks use AI to suggest strategies

to human portfolio managers. Over time, as confidence in AI’s financial acumen grows and interpretability

improves, we may see more autonomous economic agents (e.g. AI managing a sovereign wealth fund with

minimal human input). But given economic systems are socio-technical with chaotic elements, a human in

the   loop   to   handle   off-model   events   (market   crashes,   political   shifts)   is   likely   to   remain   crucial   for   the

foreseeable future.

22

Strategy Games and Competitive Strategy

In zero-sum competitive environments with well-defined rules, AI has decisively surpassed humans since

2015. AlphaGo and AlphaZero ended human dominance in Go and chess
bluffed poker pros

; Libratus and Pluribus out-
; AlphaStar and OpenAI Five achieved or exceeded pro-tier in StarCraft II and

10

20

36

1

Dota 2

33

30

. These are profound results: they show that when a strategic domain can be simulated or

learned from data, AI’s superior search and learning capacity will eventually outperform top human

intuition   and   experience.   Common   threads   include:   extensive   self-play   training,   use   of   deep   neural

networks to approximate huge strategy spaces, and iterative improvement loops that iron out weaknesses

(something humans cannot do at similar scale or speed). Particularly in games: -  Tactical precision: AI

executes strategies without mistakes or fatigue (e.g., micro-managing units in StarCraft, or calculating exact

probabilities in poker). -  Strategy innovation: AI agents often find counterintuitive strategies (AlphaGo’s

novel moves

77

, Pluribus’s unconventional bluffing frequencies

2

) that surprise humans and sometimes

elevate   theory   in   those   games.   -  Adaptation:   Through   self-play,   AI   learns   to   adapt   to   a   wide   range   of

opponent styles; humans, with more limited experience, can be caught off guard by rare tactics. Thus, in the

domain of competitive strategy under fixed rules, the human advantage has essentially vanished – AI systems

are the “champions” now in many games.

Translating this to business strategy: if we create a sufficiently faithful “game” of a market or negotiation, AI

could find winning strategies as well. We saw a hint of that in the Cambridge simulation where AI found

profit-maximizing plans better than humans

4

, and in AI Economist where AI found better policies than a

standard human approach

3

. However, real business lacks fully fixed rules and often isn’t zero-sum (there

are win-win possibilities and complex value judgments). That said,  where competition approximates a

game, AIs can excel. For example, bidding strategies in auctions (a competitive game) are now often AI-

driven, and AIs have beaten humans in simulated negotiation tournaments. The major difference: human

strategic behavior in business can be influenced by emotion, reputation, and irrational factors, which pure

game AIs don’t account for. That’s where humans still exploit AI or foresee issues an AI might miss (similar

to  how  a  pro  poker  player  might  exploit  a  known  pattern  in  an  AI’s  play,  at  least  until  it  self-corrects).

Summarily,   in  structured   strategy,   AIs   have   a   clear   edge   in   calculation,   consistency,   and   breadth   of

experience (from self-play); humans excel if the domain requires intuition beyond the model or morale/

leadership aspects.

Policy and Negotiation

This domain is nascent for AI. Diplomacy via CICERO and debate via Project Debater are groundbreaking, as

they involve  communication and strategic reasoning with humans  rather than just against them. The

results show: -  AI is now capable of meaningful natural language negotiation. CICERO held its own

among human Diplomacy players, indicating AI can engage in multi-party agreements, persuasion, and

even   subterfuge   at   a   human   level

49

48

.   It   used   polite,   context-aware   language   and   maintained

consistency with its strategic intentions. This was a leap because it required theory-of-mind – modeling

others’   beliefs   –   combined   with   language   generation.   The   success   in   a   game   context   suggests   AI

negotiators   could   soon   tackle   simpler   real   negotiations   (e.g.   automated   bargaining   in   e-commerce).   -

Humans still have the upper hand in open-ended persuasion and complex argumentation. Project

Debater showed that while AI can marshal facts and make logical points, it lacks the emotional and creative

flair of human debaters

47

. Humans prevailed overall in persuasion. This likely holds in negotiations too:

humans excel at reading subtle cues, building trust, and creatively reframing issues – areas where AI is

limited   (especially   when   stakes   involve   human   emotions   or   morality).   CICERO’s   success   came   in   a

23

constrained,   impersonal   context   (anonymous   online   games).   In   face-to-face   negotiations   or   in   building

long-term   diplomatic   relations,   current   AIs   would   likely   falter   –   they   have   no   genuine   empathy   or

understanding   of   cultural   context   unless   it’s   in   their   data   pattern,   and   missteps   can   break   trust

permanently. - AI can be superhuman in information recall and consistency. Debater could bring in far

more evidence than a person could memorize,  and CICERO  never  forgot a  deal or  message (unless by

design to be deceptive). This gives AI an edge in any negotiation or policy discussion where factual breadth

matters. It won’t miss technical details or contradict itself unintentionally – advantages in complex talks (like

trade negotiations with thousands of line items, where an AI could keep track of all). - Ethical and safety

concerns: These studies raise the point that AI negotiators could be “too effective” at influencing humans,

potentially deceiving or manipulating in unethical ways

53

. Kentaro Toyama’s comment that CICERO-like

tech is “super scary” because of how it fools humans underscores the need for oversight

78

. In current

form, humans would be wise to have AIs as assistants in negotiations rather than free agents, to ensure

values and ethical boundaries are respected.

Overall, in policy design, AI is a powerful optimizer (as seen in AI Economist). In policy negotiation, AI is

rapidly   improving   but   is   not   yet   superior   to   skilled   human   diplomats   or   leaders,   largely   due   to   the
inherently human elements (emotion, historical context, leadership legitimacy). We can imagine future AIs

assisting in real-time during negotiations – whispering suggestions to human negotiators or even taking

part in low-stakes agreements – but a “billionaire AI” cutting deals entirely on its own with other humans or

AIs would demand a level of trust we’re far from. Humans remain in charge here, with AI as a tool that can

bolster their effectiveness.

Where AI Outperforms vs Where Humans Retain Advantage

Putting it together: - AI Outperforms in: - Data-intensive analysis and optimization: Any scenario heavy

on   data   crunching,   pattern   recognition,   or   combinatorial   search   (games,   scheduling,   pricing,   resource

allocation).   AI’s   speed   and   precision   shine,   beating   human   error-prone,   slower   analysis.   E.g.,   tax   policy

optimization

3

, micro-decisions in business sims

4

, balancing complex game states in Go or StarCraft

33

. - Consistency and scale: AI doesn’t tire or get distracted, making it superior in consistent execution of

strategy   (AlphaGo   never   blunders   due   to   fatigue;   an   AI   algorithm   trader   doesn’t   need   sleep   and   can

monitor markets 24/7). In fast-paced or large-scale environments (like high-frequency trading, real-time

pricing   adjustments),   humans   simply   can’t   keep   up.   -  Novel   strategy   discovery:   AIs   often   explore

unconventional approaches free of human bias, sometimes finding creative solutions (AlphaGo’s move 37 in

game 2 vs Lee Sedol was famously novel; AI Economist found non-linear tax schemes). This can outperform

human adherence to traditional thinking. - Multi-variable reasoning: In contexts with many variables and

constraints (multi-player games, complex logistical planning), AI can juggle these better than humans, who

might oversimplify. Pluribus handling 5 opponents simultaneously

36

 or OpenAI Five coordinating 5 units

are examples. -  Rapid adaptation (within trained distribution): AIs can quickly adjust tactics if within

what they’ve trained for – e.g., AlphaStar adjusting when it recognized a known rush strategy from the

opponent, or Libratus recalculating strategy nightly to plug leaks

22

. Humans adapt too but are slower and

limited by cognitive biases.

•

Humans Retain Advantage in:

•

Unpredicted situations and generalization: When the scenario goes out-of-distribution (a black

swan event in a market, a new rule in a game, a novel negotiation topic), humans can use intuition

and analogy to navigate it, whereas AI may be clueless. The Cambridge experiment’s black swan

24

result

59

 exemplifies this – humans coped better with shocks. Likewise, if StarCraft’s rules changed

suddenly, human pros might improvise better initially than AlphaStar (until retraining).

•

Long-term planning with open-ended goals: Humans are better at keeping a broader vision over

months/years, integrating changing goals or values. AI tends to optimize a fixed objective and can

struggle if the mission evolves or if there’s no clear metric. Strategic leadership (setting vision,

culture) is still a human forte.

•

Empathy, ethics, and trust: Involvement of people means feelings and moral considerations.

Humans can build trust, inspire teams, sense unspoken issues – AI cannot genuinely empathize or

handle moral trade-offs without explicit instructions. Debater couldn’t sway emotion, whereas a

human can tell a moving story. In negotiation, human trust networks and reputations carry weight

that an AI doesn’t have.

•

Creativity and innovation: Outside rule-bound domains, human creativity (inventing a new

product, a new business model) remains superior. AIs innovate within confines of data (AlphaGo’s

creativity in Go is real, but it operates within Go’s rules). For divergent thinking or paradigm shifts,

humans still lead – AIs won’t propose entirely new games to play or unasked questions (at least not

intentionally – random generation isn’t the same as purposeful creativity).
Common sense and qualitative judgment: Humans excel at integrating qualitative factors – e.g.,

•

understanding a customer’s irrational preferences or an employee’s motivation. AI often lacks this

“soft” understanding. A CEO must sometimes make a gut call that defies the spreadsheet (like

sacrificing short-term profit for brand integrity) – humans can do this guided by experience and

values; an AI optimizing profit might not.

•

Accountability and strategic risk management: Humans inherently think about accountability

(their own job, reputation, legal implications). AI might take strategies that look good in simulation

but if wrong, could be catastrophic (with no skin in the game, AI doesn’t feel risk). Humans often

moderate decisions to avoid worst-case scenarios – a caution that is sometimes wise.

In   summary,  AI   currently   dominates   bounded,   data-rich   challenges  and   performs   tasks   requiring

superhuman consistency or computational depth. Humans dominate unbounded, high-level, and deeply

social   challenges.   The   frontier   is   moving:   tasks   that   were   unbounded   (like   language   negotiation)   are

increasingly tackled by AI with some success, but full human-level generality remains out of reach as of

2025.

Trajectory & Forecast Toward Billionaire-Level

Autonomy

The timeline of progress (2015–2025) reveals an acceleration in AI capabilities and autonomy, suggesting

a trajectory where AI systems become increasingly competent in roles that were exclusively human. We

observe a pattern: AI first conquers highly structured domains (games, well-defined tasks), then rapidly

moves into messier real-world domains (business strategy, language negotiation) once the core techniques

(deep   learning,   self-play,   large   language   models)   mature   and   compute   power   scales   up.   Key   enabling

trends and performance improvements include:

•

Exponential growth in model scale & computing: AlphaGo’s triumph was enabled by deep neural

nets and TPUs; by 2023, GPT-4 (with hundreds of billions of parameters) can digest and reason over

vast business data, something impossible in 2016. The compute used in AI training doubled every

3.4 months through late 2010s, a trend fueling breakthroughs. This scaling has a direct payoff: larger

25

models and more self-play iterations yield qualitatively new abilities (e.g., GPT-3 vs GPT-4 differences

in   reliability).   If   this   continues   (through   specialized   hardware,   better   algorithms,   or   just   more

investment), by 2030 we could have AI models with virtually encyclopedic knowledge and refined

reasoning approaching expert-human level across many domains simultaneously.

•

Integration   of   learning   paradigms:   Initially,   the   big   wins   were   in   self-play   RL   (games)   and

supervised learning (vision, language). Now we see hybrid systems (e.g., CICERO combining RL with

LLMs, or the AI Economist combining multi-agent RL with economic simulation, or GPT-4 being used

for   decision-making   in   Cambridge’s   sim).   The   ability   to   integrate   language   understanding   with

strategic planning is a game-changer – it means AI can operate in the same medium humans do

(natural language) while leveraging algorithmic precision. This convergence will enable AI agents to

participate in human workflows seamlessly, which is necessary for an autonomous “billionaire AI”

agent that must negotiate deals, read legal documents, and so forth. Future systems (beyond 2025)

are   likely   to   pair   advanced   LLMs   with   specialized   decision   planners,   yielding   agents   that   both

understand and act in the business world.

•

Autonomy architecture advancements: Tasks like Vending-Bench highlight weaknesses, but also

drive research into solutions (like long-context transformers, external memory, or agent frameworks

like AutoGPT). We expect significant improvements in long-horizon autonomy in the next few years

–   early   experiments   with   agents   that   can   execute   multi-step   goals   (e.g.   GPT-based   “AutoGPT”

instances on the internet) are already emerging in 2023. While those are rudimentary, they point to

systems that can manage multi-day or multi-month projects. By combining improved memory with

iterative planning, AI may overcome the coherence problem by 2030 or so.

•

Human-AI collaboration know-how: The studies advise hybrid models of leadership (AI handling

analysis, humans focusing on judgment)

66

. In practice, as more businesses experiment with AI in

executive functions, best practices will develop. For example, a possible trajectory: within 5 years,

many companies have AI advisors in board meetings analyzing trends; within 10 years, some firms

let   AI   make   limited   strategic   decisions   autonomously   (e.g.,   pricing   optimization   or   supply   chain

adjustments) with humans overseeing. Each success builds trust, each failure teaches safeguards. A

fully   autonomous   company   (an   AI   CEO   with   minimal   human   board   oversight)   might   start   as   an

experiment in a sandbox jurisdiction or a small private enterprise in the 2030s. If that succeeds (and

makes serious money), it will catalyze adoption.

•

Continuous learning and adaptation: Future AI agents will likely be online learners – updating their

models on the fly as new data comes (like how a human CEO learns each quarter). The current

paradigm   of   train-then-deploy   will   shift   to   continuous   learning   (with   caution   to   avoid   drift).   This

means   a   “billionaire   AI”   could   keep   getting   better   at   its   domain,   potentially   outpacing   human

improvement which is bounded by lifespan and cognition. For instance, an AI trading firm could

adapt to new market regimes overnight, whereas human organizations take weeks or months.

Combining   these   trends,   what   is   the   outlook   for   an   AI   agent   reaching   “billionaire”   status   (meaning   it

autonomously creates enormous economic value, comparable to a top entrepreneur or investor)?

10-year horizon (by 2035): We project it’s plausible (perhaps a 20% probability) that a proto-billionaire AI

emerges. This might be an AI-driven hedge fund or trading algorithm that, with minimal human input,

accumulates large profits (some quant funds are already highly automated – an AI could take that further

26

and run strategies end-to-end). Or it could be an AI-led startup (e.g. providing an AI service that becomes

extremely valuable) where the AI handles product development and operations, with humans only formally

in   the   loop.   Technologically,   by   10   years   we   likely   have   AIs   that   can   manage   quite   complex   tasks:   for

example, an AI that can read market news, execute trades, negotiate simple contracts (via APIs) and do so

at scale. Socially and legally, 10 years is short for fully autonomous companies to be accepted, but limited

trials could happen. We might see “virtual CEOs” appointed in small firms under human legal guardianship.

However, broad replacement of human execs is unlikely by 2035 due to trust and regulatory inertia. More

common will be AI “co-pilots” for executives becoming standard.

20-year horizon (by 2045): If AI progress continues without major roadblocks, by 20 years the chances

increase  (50%  or  more)  that  true  billionaire  AI  agents  exist.  By  2045,  AI  systems  could  pass  not  just

cognitive bar exams (they already do) but demonstrate reliability and strategic acumen in real markets

consistently. We may have witnessed some AI-managed enterprises that outcompete human-led ones in

efficiency.   Enabling   trends   like   quantum   computing   or   fundamentally   new   ML   paradigms   (if   they

materialize) could exponentially boost AI decision-making power. With generational turnover, society might

be   more   amenable   to   letting   an   AI   take   larger   autonomous   roles,   especially   if   evidence   shows   they
outperform humans ethically and economically. It’s conceivable by 2045 there are corporations or funds

effectively run by AI (with human oversight mostly pro-forma), and some of those AIs may have amassed

wealth (for their shareholders or controlling entities) on the order of billions. Whether we label the AI itself a

“billionaire”  is  semantic  –  more  likely  we’d  say  the  AI  made  its  human  owners  billionaires.  But  one  can

imagine   advanced   AI   agents   perhaps   owning   shares   or   negotiating   equity   for   themselves   (sci-fi   as   it

sounds, legal personhood for AI could be debated by then).

Never or unknowable: There is a non-zero chance (perhaps 10%) that we  never  see a truly autonomous

billionaire AI. This could be due to external factors – e.g., global regulations might strictly limit AI autonomy

for safety reasons, or economic structures might change (if, say, AI triggers a post-scarcity economy or a

major societal shift that deemphasizes capital accumulation). It’s also possible there are fundamental limits

to AI in the open world that we haven’t yet discovered – maybe certain creative and emotional intelligences

needed for top entrepreneurship can’t be learned by machines easily. However, given the momentum of

current   progress,   “never”   seems   unlikely   unless   humanity   itself   steps   back   from   using   AI   that   way.

“Unknowable” is a fair category – beyond 20 years, predictions are shaky. AI could plateau or we might

encounter diminishing returns. But as of 2025, the trajectory is still steeply upward.

It’s   also   important   to   consider  qualitative   changes  that   might   occur   en   route:   If   AI   does   approach

billionaire-level   capability,   that   would   dramatically   alter   markets.   The   first   AI   agents   making   major

autonomous economic moves could cause both huge opportunity and disruption – e.g., financial markets

dominated   by   superintelligent   algos   might   be   very   volatile   or   oddly   efficient,   leaving   less   room   for

traditional investors. Companies with AI CEOs could outcompete others or, conversely, face backlash from

consumers/governments not wanting impersonal control. These factors will influence adoption.

Trajectory summary: In the late 2010s, we saw AI exceed human specialists in bounded tasks. Early 2020s,

AI began entering general professional arenas (coding assistants, strategy simulations). Late 2020s should

bring   robust   autonomous   agents   handling   multi-faceted   professional   tasks   under   supervision.   By   the

2030s, likely one or more instances of highly autonomous AI-driven entities achieving dramatic success

(technologically   possible,   policy   permitting).   That   will   test   society’s   comfort   –   if   outcomes   are   positive

(growth,   efficiency,   minimal   downsides),   it   will   accelerate   broader   deployment.   If   negative   (e.g.,   an   AI

27

triggers a flash crash or makes an inhumane decision that causes scandal), that might slow the trajectory

with stricter controls and insistence on human oversight.

Our forecast leans optimistic in capability:  AI will be technically ready for billionaire-level autonomy

within ~20 years. The probability we assign: - Within 10 years (~2035): low (~20%) that an AI agent, largely

autonomous,   creates   billionaire-level   value.   Limited   to   niche   domains   (likely   finance).   -   Within   20   years

(~2045):   moderate-to-high   (~50%)   that   AI   agents   achieve   this   in   multiple   domains,   potentially   openly

recognized as making top-level decisions. - Never: low (~10%) given current knowledge – barring a global

moratorium or unforeseen scientific barrier. - Unknowable remainder (~20%): acknowledges uncertainty in

socio-political response and unknown unknowns in AI progress (could be sooner or later depending on

breakthroughs or setbacks).

Barriers & Enablers

Realizing   a   billionaire-level   autonomous   AI   agent   is   not   just   a   technical   question;   it   intersects   with

economics, law, and ethics. Here we outline key barriers and enablers:

Technical   Barriers:  Current   AI   systems,   as   seen,   have   issues   with   long-term   reliability

(Vending-Bench failures

5

), generalization to truly novel situations, and understanding of

human   nuance   (Debater’s   lack   of   emotional   appeal

47

).   To   trust   an   AI   with   an   entire

enterprise, these must be overcome. We need advances in  robust AI: e.g., better memory

architectures   (to   prevent   forgetting   or   derailment),   causal   reasoning   (so   AI   can   handle

situations it hasn’t explicitly seen by reasoning from first principles), and alignment (so AI’s

objectives   remain   in   line   with   human   goals   even   as   it   operates   autonomously).   Ensuring

safety  is paramount – a misaligned powerful AI in a business could cause economic havoc

(imagine an AI CEO maximizing profit with no regard for law or ethics). Technical research in

alignment (the AI following true human intent, not proxy metrics) is an essential enabler for

deployment. If unsolved, it’s a barrier: no one will deploy an AI CEO that might decide the

best way to maximize profit is, say, illegal or harmful. On the flip side, technical progress in

interpretability (being able to explain AI decisions) and verification (formally checking AI plans

against safety constraints) would enable greater trust and legal acceptance.

Economic and Market Barriers: For an AI to become effectively a billionaire agent, it must

operate in markets that allow it.  Competition  could be a barrier: if many AIs enter finance,

their profits could zero-sum out. For instance, if every hedge fund uses similar AI, beating the

market consistently might become harder – a “richest AI” arms race might yield diminishing

returns. There’s also the barrier of capital and access: current AIs don’t have legal agency to

own   assets   or   sign   contracts.   They   typically   act   as   tools   for   human   owners.   Unless   laws

change (see below), an AI accumulating wealth will do so under a corporate or individual’s

name. So an “autonomous” billionaire AI might still be legally tied to a human or company

that provided it capital and authorization. However, enablers on the economic side include the

massive efficiency gains AI can provide – companies might push AIs into higher roles to gain

edge (as evidenced by the Cambridge study hints that AI could disrupt consulting

65

). If one

company benefits greatly from an AI executive (higher profits, faster decisions), competitors

will   feel   pressure   to   follow,   spurring   wider   adoption   –   a   positive   feedback   enabling   AI

autonomy.   Another   barrier:  consumer   and   employee   acceptance.   If   stakeholders   don’t   trust

28

decisions made by an AI (for instance, employees might resist directives from an impersonal

algorithm, or consumers might avoid a brand run by AI due to lack of human connection),

that could slow or limit autonomy. Society tends to value human elements in leadership; a

transition period with hybrid models might be necessary to prove AI can lead responsibly and

effectively.

Legal   and   Regulatory   Factors:  Currently,   AI   systems   have   no   legal   personhood.   Laws

require human accountable officers and directors for companies. For an AI to formally run a

business, corporate law would need to evolve (e.g., allowing an AI to be an officer or to make

binding decisions). We might see intermediate steps: regulatory sandboxes where firms can

test AI decision-makers, or a requirement that a human is nominally in charge even if AI does

the work. There’s also liability: If an AI CEO makes a decision leading to harm or fraud, who is

liable? The company? The developers? This lack of clarity is a major barrier. Regulators may

impose strict oversight on AI in critical roles (like how algorithmic trading is monitored to

prevent flash crashes). If early uses of AI execs cause publicized failures, regulations might

sharply   limit   AI   autonomy   (like   requiring   explainability   or   emergency   stop   mechanisms
controlled by humans). On the other hand, if AI-driven companies show strong performance

and  good compliance, regulators might gradually open up. Another legal enabler would be

frameworks assigning AIs some legal status or clarifying liability – for instance, treating an AI

as   an   “ultra-risky   employee”   where   the   company   assumes   responsibility   for   its   actions.

International competition in AI could also drive regulatory easing: if one jurisdiction (say a

certain   country)   allows   AI-run   companies   and   they   prosper,   others   might   follow   to   stay

competitive. Data privacy laws and anti-bias regulations are additional concerns: AI decisions

might inadvertently discriminate or misuse data, incurring legal issues. Ensuring AIs abide by

laws  (financial  regulations,  labor  laws,  etc.)  is  critical  –  ironically  an  AI  might  be  better  at

strictly following rules than humans, but only if those rules are encoded as constraints in its

training or operation.

Ethical and Societal Considerations:  Ethically, handing over major decisions to machines

raises concerns of fairness, transparency, and the value of human agency. Society may resist

if   it   feels   like   “soulless   machines”   are   in   control   of   livelihoods.   One   barrier   is   the  loss   of

human jobs and expertise  – if AI runs everything, what do humans do? There could be a

backlash similar to automation fears in labor: we might see movements to “keep humans in

the loop” akin to how some countries ban fully autonomous lethal weapons. Ensuring AI-led

growth   benefits   society   at   large   (not   just   making   the   owners   of   AI   richer)   is   an   ethical

imperative to prevent unrest. Another ethical aspect is how AI will make moral choices – e.g.,

an AI CEO might face a dilemma: recall a product that has a small defect at huge cost or cover

it up? Humans make such calls with a moral compass (varying quality, but still). Will AI default

to utilitarian calculations? This worries people. Building ethical reasoning into AI or at least

bounding   them   by   ethical   guidelines   is   an   enabler;   a   barrier   is   the   difficulty   of   encoding

nuanced values. There’s also the concept of human dignity and purpose: some argue even if

AI   could   do   everything,   humans   should   still   have   a   meaningful   role   in   directing   our

institutions. This philosophical stance might influence policy (for instance, requiring that final

authority   lies   with   a   human   board).   Yet,   consider   enablers:   if   AI   leadership   leads   to

demonstrably   more   ethical   outcomes   (e.g.,   less   corruption,   more   objective   and   inclusive

decisions), stakeholders might prefer AI. For instance, an AI judge might be more consistent

and unbiased than a human judge – in theory leading to fairer justice. If similarly an AI CEO

29

always follows ethical guidelines (no cheating or emotional outbursts), it might earn public

trust over erratic human CEOs. To enable that, AI governance structures must be set so that

these systems act transparently and align with societal values (perhaps even more strictly

than human execs, who sometimes behave unethically). Ethically, including diverse values in

AI   training   (to   avoid   e.g.   only   profit   motive)   will   be   key.   The   Salesforce   AI   Economist,   for

example, was explicitly trying to optimize social welfare (equality + productivity)

3

, not just

GDP – if AI can be taught to consider broad stakeholder outcomes, that could make them

more ethically palatable leaders than single-minded humans or those driven by short-term

gains.

In   summary,  barriers  to   a   billionaire   AI   agent   are   significant   but   appear   surmountable   over   time:   -

Technical reliability and alignment issues – being actively researched. - Legal frameworks – likely to lag, but

can adapt if pressure and evidence build. - Social acceptance – depends on proving that AI can lead to good

outcomes for most people and not just profit. - Ethical design – must progress so AI decisions respect

human values and rights.

Enablers  that   could   accelerate   it:   -   Demonstration   projects   (like   more   Cambridge-style   experiments   in

industries)   showing   AI’s   superior   performance   under   oversight.   -   A   competitive   race   (countries   or

companies)   that   incentivizes   empowering   AI   as   much   as   possible,   breaking   through   inertia.   -   Gradual

successful integration (AI CFOs then CEOs, etc.) that normalizes the concept. - Continued rapid AI innovation

making these agents obviously better than average human leaders in many tasks, making it economically

irrational not to use them.

Conclusion

In the last decade, AI systems have moved from laboratory curiosities beating humans in board games to

active participants in complex business and policy simulations. We synthesized evidence that: - AI already

rivals   or   exceeds   human   experts   in   numerous   strategic   tasks  (from   game   play   to   tax   policy

optimization), often achieving superhuman performance within bounded conditions

1

3

. - Humans still

provide   critical   strengths  in   creativity,   adaptability,   ethical   judgment,   and   managing   novel   situations,

which current AI lacks

47

61

. - Hybrid human-AI approaches are currently the most effective: AI provides

analysis and consistency, humans provide direction and values

66

.

Looking ahead, the trajectory of AI capabilities suggests that the gap in higher-level autonomy will continue

to close. Each domain we reviewed shows a pattern of initial AI success in narrow scopes, followed by rapid

expansion into more complex scopes as techniques improve and more data becomes available. If these

trends   hold:   -   We   may   witness   early   forms   of  autonomous   AI   economic   agents  within   the   next   two

decades – for instance, AI-run investment funds or fully automated enterprises in controlled sectors. Their

success will depend not just on raw intelligence, but on gaining trust through reliability and alignment with

human interests. - Achieving a  “billionaire AI agent”  is as much a socio-economic challenge as a technical

one.   Technically,   it’s   increasingly   feasible;   socially   and   legally,   it   requires   rethinking   frameworks   of

accountability and ownership. - In an optimistic scenario, AI autonomy could lead to enormous efficiencies

and innovations, augmenting human decision-making and possibly tackling global challenges with rigor

and impartiality (imagine AI policy advisors optimizing responses to climate change or poverty with fine-

grained strategies humans might miss). - In a cautious scenario, strong oversight and phased integration

will   be   necessary   to   avoid   pitfalls   (like   misaligned   incentives   or   loss   of   human   agency).   The   studies

30

underscore   that  blind   spots   and   failure   modes   exist,   so   a   future   with   AI   executives   will   need   robust

guardrails and perhaps a new kind of corporate governance where AI and human trustees collaborate.

Our Executive Summary provided probability estimates for seeing a true billionaire-level AI agent: ~20%

within 10 years, ~50% within 20 years, acknowledging uncertainties beyond. It is possible such an entity

never  fully materializes under current paradigms (~10% chance), but given the pace of progress, we lean

towards it emerging in some form if society allows.

In conclusion, AI systems are on a clear trend towards mastering complexity and demonstrating autonomy

in domains once thought uniquely human. Humans still hold crucial advantages, but those are narrowing to

realms of emotion, morality, and extreme uncertainty. Whether an AI will literally sit atop a Fortune 500

company as CEO or control a multi-billion-dollar fund on its own by mid-century remains uncertain, but the

foundations are being laid now. The studies we reviewed show that piece by piece, AI is acquiring the skills

needed: strategic planning, negotiation, operational management, and learning from experience. Bridging

those pieces into an integrated, trustworthy “billionaire AI” will be the challenge of the next 10–20 years,

involving not just engineering, but wise governance and ethical stewardship. The competitive and financial

incentives to do so are enormous – where AI clearly outperforms, it will be adopted. The critical question is

ensuring that as we grant AI more autonomy and power, we also imbue it with the values and oversight

necessary to use that power for the broader good, rather than purely for profit or in ways that undermine

human dignity and societal stability.

Probability Forecast: Based on current evidence and trends, we assign roughly a 50% likelihood that an AI

agent   will   attain   billionaire-level   autonomy   (creating   tremendous   economic   value   with   minimal   human

input) within the next 20 years. There is about a 20% chance of this occurring much sooner (within 10 years,

in a limited domain like algorithmic trading). There remains perhaps a 10% chance it never occurs under

present paradigms, due either to insurmountable technical alignment issues or deliberate human policy

decisions   to   restrict   AI   autonomy.   The   remaining   uncertainty   (~20%)   acknowledges   unpredictabilities   in

innovation   and   global   response.   In   essence,   barring   a   drastic   slowdown   or   policy   clampdown,   the

emergence of highly autonomous, economically dominant AI agents by the mid-21st century appears more

likely than not.

Reference List

1.

Chouard, T. (2016). “The Go Files: AI computer wraps up 4-1 victory against human champion.” Nature

News (15 March 2016). [AlphaGo’s 4–1 defeat of Lee Sedol, marking the first time an AI beat a top Go

master]

1

77

2.

Queensland   Brain   Institute.   (2017).  “Google   AlphaGo   Zero   masters   the   game   in   three   days.”  (Press

release, 23 Oct 2017). [AlphaGo Zero used self-play to beat the original AlphaGo 100–0 after 3 days of

training]

10

6

3.

Sandholm,   T.   &   Brown,   N.   (2017).  “Libratus:   The   Superhuman   AI   for   No-Limit   Poker.” IJCAI   2017

Proceedings, pp. 5226-5233. [Architecture with three modules enabling Libratus to beat top humans

in heads-up poker]

12

11

31

4.

Carnegie   Mellon   University   News.   (2017).  “AI   beats   top   poker   pros.”  (Press   Release,   31   Jan   2017).

[Libratus defeated four human poker champions over 120k hands, with a statistically significant win

margin of $1.77M in chips]

79

24

5.

Phys.org / AFP. (2019).  “AI program beats pros in six-player poker—a first.”  (12 July 2019). [Pluribus

achieved superhuman performance in 6-player Texas Hold’em, marking the milestone of AI in multi-

party imperfect-information games]

39

44

6.

Kennedy, M. (2019). “Bet On The Bot: AI Beats The Professionals At 6-Player Texas Hold 'Em.” NPR News

(11 July 2019). [Facebook/CMU’s Pluribus AI consistently beat elite poker pros, with an estimated win

rate ~$1k/hour against them]

38

7.

OpenAI.  (2019).  “OpenAI  Five  defeats  Dota  2  world  champions.”  (Blog,  15  April  2019).  [OpenAI  Five

became the first AI to defeat the reigning world champion Dota 2 team (OG) in back-to-back games,

demonstrating AI prowess in a complex esports game]

30

8.

Vinyals, O.  et al.  (2019).  “AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement

learning.” DeepMind   Blog  (30   Oct   2019).   [AlphaStar   achieved   Grandmaster   rank   (top   0.2%)   on

StarCraft II’s online ladder for all races, under human-like game constraints]

33

9.

Slonim,   N.  et   al.  (2021).  “An   autonomous   debating   system.” Nature,   591:379–384.   [IBM’s   Project

Debater architecture and evaluation; while the AI could debate on many topics and provide facts,

humans still prevailed in overall persuasiveness and the system highlighted the gap between game

challenges and human debate]

47

10.

Metz, C. (2022). “Meta’s Cicero is first AI to master Diplomacy, the strategy game that requires human-like

negotiation.” The Washington Post (1 Dec 2022). [Meta AI’s CICERO agent placed in the top 10% of

human   players   in   an   online   Diplomacy   league   by   combining   strategic   planning   with   natural

language negotiation, often fooling human players into thinking it was human]

49

48

11.

Zheng,   S.  et   al.  (2022).  “The   AI   Economist:   Taxation   policy   design   via   two-level   deep   multiagent

reinforcement   learning.” Science   Advances,   8(18):   eabk2607.   [Salesforce’s   AI   Economist   learned

dynamic tax policies that improved social welfare (balancing equality and productivity) more than

standard tax systems, showcasing AI outperforming human-crafted economic policy in simulation]

3

54

12.

Mudassir, H. et al. (2024). “AI Can (Mostly) Outperform Human CEOs.” Harvard Business Review (Sept

26,   2024).   [Experiment   at   Cambridge   Judge   Business   School   where   a   GPT-4-based   AI   competed

against hundreds of humans in a realistic automotive industry simulation; AI excelled at data-driven

decisions (boosting profit and market share) but struggled with unforeseen disruptions and cannot

replace the full human role, suggesting a hybrid future for leadership]

67

61

13.

Backlund,   A.   &   Petersson,   L.   (2025).  “Vending-Bench:   A   Benchmark   for   Long-Term   Coherence   of

Autonomous Agents.”  (arXiv preprint arXiv:2502.15840, Feb 2025). [Introduces a simulated vending

machine   business   to   stress-test   LLM-based   agents   over   >20   million   token   runs;   top   models

sometimes outperformed a human baseline in profit, but all suffered from lapses (mis-scheduled

32

orders,   forgotten   context,   nonsense   loops)   in   at   least   some   runs,   underscoring   current   limits   in

sustained autonomous coherence]

69

71

1

7

77

SAAS Berkeley

https://saas.studentorg.berkeley.edu/rp/an-introduction-to-go-alphago-and-quantifying-go-gameplay

2

36

37

39

40

41

42

43

44

AI program beats pros in six-player poker—a first

https://phys.org/news/2019-07-ai-pros-six-player-pokera.html

3

54

Science Advances Publishes AI Economist Research on Improving Tax Policies With RL

https://www.salesforce.com/blog/ai-economist-science-advances/

4

55

58

59

60

61

62

63

65

66

67

68

New research: Human vs AI CEOs - News & insight - Cambridge

Judge Business School

https://www.jbs.cam.ac.uk/2024/new-research-human-vs-ai-ceos/

5

69

70

71

72

73

74

75

76

Vending-Bench: A Benchmark for Long-Term Coherence of Autonomous

Agents

https://arxiv.org/html/2502.15840v1

6

10

26

27

Google AlphaGo Zero masters the game in three days - Queensland Brain Institute -

University of Queensland

https://qbi.uq.edu.au/blog/2017/10/google-alphago-zero-masters-game-three-days

8

9

After 3 Losses, Master Go Player Scores A Win Against Computer : The Two-Way : NPR

https://www.npr.org/sections/thetwo-way/2016/03/13/470284113/after-three-losses-master-go-player-scores-a-win-against-

computer

11

12

13

14

15

21

23

Libratus: The Superhuman AI for No-Limit Poker

https://www.ijcai.org/proceedings/2017/0772.pdf

16

17

18

19

20

22

24

25

79

Carnegie Mellon Artificial Intelligence Beats Top Poker Pros - News -

Carnegie Mellon University

https://www.cmu.edu/news/stories/archives/2017/january/AI-beats-poker-pros.html

28

29

31

Takeaways from OpenAI Five (2019) | by Jeffrey Shek | TDS Archive | Medium

https://medium.com/data-science/takeaways-from-openai-five-2019-f90a612fe5d

30

OpenAI Five defeats Dota 2 world champions | OpenAI

https://openai.com/index/openai-five-defeats-dota-2-world-champions/

32

33

34

35

AlphaStar: Grandmaster level in StarCraft II using multi-agent reinforcement learning -

Google DeepMind

https://deepmind.google/discover/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/

38

Poker Bot Beats The Professionals At 6-Player Texas Hold 'Em : NPR

https://www.npr.org/2019/07/11/740661470/bet-on-the-bot-ai-beats-the-professionals-at-6-player-texas-hold-em

45

An IBM AI Debates Humans--but It's Not Yet the Deep Blue of Oratory

https://www.scientificamerican.com/article/an-ibm-ai-debates-humans-but-its-not-yet-the-deep-blue-of-oratory/

46

IBM Project Debater - AIAAIC

https://www.aiaaic.org/aiaaic-repository/ai-algorithmic-and-automation-incidents/ibm-project-debater

33

47

An autonomous debating system for Nature - IBM Research

https://research.ibm.com/publications/an-autonomous-debating-system

48

49

51

52

53

78

Meta’s new AI is skilled at a ruthless, power-seeking game - The Washington Post

https://www.washingtonpost.com/technology/2022/12/01/meta-diplomacy-ai-cicero/

50

Human-level play in the game of Diplomacy by combining language ...

https://www.science.org/doi/10.1126/science.ade9097

56

57

64

How OpenAI and GPT 4o outperformed human CEOs | Hamza Mudassir posted on the topic |

LinkedIn

https://www.linkedin.com/posts/hamzamudassir_ai-can-mostly-outperform-human-ceos-activity-7245061506386604032-k9cl

34

