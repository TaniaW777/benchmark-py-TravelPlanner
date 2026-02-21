# Framework Comparison: OpenSymbolicAI vs LangChain vs CrewAI

Head-to-head comparison on the [TravelPlanner benchmark](https://osu-nlp-group.github.io/TravelPlanner/) (ICML 2024) — 45 travel planning tasks across easy, medium, and hard difficulty levels. Same model, same tools, same evaluation. Only the framework differs.

## The Bottom Line

```
                        OpenSymbolicAI    LangChain    CrewAI
                        ──────────────    ─────────    ──────
  Pass Rate                    100%         77.8%      73.3%
  Tokens per Task            13,936        43,801     81,331
  LLM Calls per Task            2.3          13.5       39.6
  Cost per Passing Task      $0.013        $0.051     $0.100
  Avg Latency                   47s           73s       124s
```

OpenSymbolicAI passes every task. The others don't — and they burn through significantly more tokens trying.

---

## The Multiplier Effect

The efficiency gap isn't incremental — it's multiplicative.

### vs LangChain

| Dimension | OpenSymbolicAI | LangChain | Factor |
|-----------|:-:|:-:|:-:|
| **Pass Rate** | 100% | 77.8% | **1.3x higher** |
| **Tokens / Task** | 13,936 | 43,801 | **3.1x fewer** |
| **LLM Calls / Task** | 2.3 | 13.5 | **5.9x fewer** |
| **Cost / Passing Task** | $0.013 | $0.051 | **4.1x cheaper** |
| **Latency** | 47s | 73s | **1.5x faster** |
| **Total Cost (45 tasks)** | $0.56 | $1.77 | **3.1x cheaper** |

### vs CrewAI

| Dimension | OpenSymbolicAI | CrewAI | Factor |
|-----------|:-:|:-:|:-:|
| **Pass Rate** | 100% | 73.3% | **1.4x higher** |
| **Tokens / Task** | 13,936 | 81,331 | **5.8x fewer** |
| **LLM Calls / Task** | 2.3 | 39.6 | **17x fewer** |
| **Cost / Passing Task** | $0.013 | $0.100 | **8x cheaper** |
| **Latency** | 47s | 124s | **2.6x faster** |
| **Total Cost (45 tasks)** | $0.56 | $3.29 | **5.8x cheaper** |

---

## Reliability Under Pressure

Pass rates diverge as task complexity increases. OpenSymbolicAI stays at 100% regardless of difficulty. LangChain and CrewAI degrade.

| Difficulty | OpenSymbolicAI | LangChain | CrewAI |
|:----------:|:-:|:-:|:-:|
| **Easy** (15 tasks) | 100% | 93.3% | 80.0% |
| **Medium** (15 tasks) | 100% | 73.3% | 80.0% |
| **Hard** (15 tasks) | 100% | 66.7% | 60.0% |
| | | | |
| **All** (45 tasks) | **100%** | **77.8%** | **73.3%** |

On hard tasks — multi-city trips with budget, cuisine, room type, and transportation constraints — LangChain drops to 67% and CrewAI to 60%. OpenSymbolicAI doesn't miss a single one.

---

## Token Efficiency Breakdown

Where the tokens go tells the story. OpenSymbolicAI spends most of its budget on plan assembly (the valuable part). LangChain and CrewAI burn tokens on retrieval overhead.

| Phase | OpenSymbolicAI | LangChain | CrewAI |
|-------|:-:|:-:|:-:|
| **Retrieval** (avg tokens) | 3,546 | 40,822 | 74,733 |
| **Assembly** (avg tokens) | 10,390 | 2,979 | 6,598 |
| **Total** | 13,936 | 43,801 | 81,331 |

OpenSymbolicAI's retrieval phase is **11.5x** more efficient than LangChain's and **21x** more efficient than CrewAI's. It gathers the same data with a fraction of the tokens because one LLM call generates Python code that makes all required API calls — no multi-turn ReAct loops, no agent-to-agent handoffs.

---

## Constraint Satisfaction

The TravelPlanner benchmark evaluates 13 individual constraints across two categories. OpenSymbolicAI achieves perfect scores on every single one.

| Category | OpenSymbolicAI | LangChain | CrewAI |
|----------|:-:|:-:|:-:|
| Commonsense (8 checks) — macro | 100% | 86.7% | 82.2% |
| Commonsense (8 checks) — micro avg | 100% | 98.1% | 97.8% |
| Hard Constraints (5 checks) — macro | 100% | 91.1% | 91.1% |
| Hard Constraints (5 checks) — micro avg | 100% | 95.6% | 95.6% |

LangChain and CrewAI achieve high micro-averages (most individual checks pass), but their macro rates drop significantly — meaning when a plan fails, it often fails multiple checks simultaneously. This is the cascade effect of incomplete data retrieval: a single missing restaurant search leads to hallucinated entity names, which fails both the "within sandbox" and "diverse restaurants" checks.

---

## Why the Gap Exists

The three frameworks use fundamentally different patterns for the same task:

| | OpenSymbolicAI | LangChain | CrewAI |
|---|---|---|---|
| **Pattern** | Code generation | ReAct agent loop | Multi-agent crew |
| **Retrieval** | 1 LLM call generates Python that makes all API calls at once | Agent decides one tool call per turn, loops until done | Researcher agent iterates with tools, hands off to planner |
| **Assembly** | 1 LLM call generates Python that builds the plan programmatically | Single LLM call to produce plan JSON | Planner agent produces plan JSON from researcher's text output |
| **LLM Calls** | 2-3 total | 10-15 total | 25-40 total |
| **Failure mode** | Deterministic — code either runs or throws a traceable error | Agent stops early, missing data leads to incomplete plans | Researcher misses tools, planner hallucinates entity names |

The core insight: **one LLM call generating code that makes 15 API calls is fundamentally more efficient than 15 separate LLM turns each making one API call.** The ReAct pattern (LangChain) and multi-agent delegation (CrewAI) both pay a per-turn overhead that compounds with task complexity.

---

## Methodology

- **Model**: `gpt-oss-120b` via Fireworks AI (same for all frameworks)
- **Dataset**: Full TravelPlanner train split — 45 tasks (15 easy + 15 medium + 15 hard)
- **Evaluation**: Official TravelPlanner constraint checks (8 commonsense + 5 hard)
- **Shared infrastructure**: All frameworks use the same `ReferenceDatabase`, search primitives, and evaluation pipeline
- **Post-processing**: All frameworks share the same deterministic field-filling step — the comparison isolates the framework's LLM interaction pattern
- **Parallelism**: 10 concurrent workers per framework
- **Source**: [osunlp/TravelPlanner](https://huggingface.co/datasets/osunlp/TravelPlanner) (ICML 2024)

### Reproduce

```bash
uv sync --extra langchain --extra crewai

uv run travelplanner-compare \
    --frameworks opensymbolicai,langchain,crewai \
    --model gpt-oss-120b --provider fireworks \
    --split train -p 10
```

---

## Per-Task Results

Every task, every framework. OpenSymbolicAI passes all 45. LangChain fails 10. CrewAI fails 12.

<details>
<summary>Click to expand full per-task breakdown</summary>

### Easy (15 tasks)

| Task | OpenSymbolicAI | | LangChain | | CrewAI | |
|------|:-:|--:|:-:|--:|:-:|--:|
| | Pass | Tokens | Pass | Tokens | Pass | Tokens |
| tp_0000 | PASS | 10,797 | PASS | 20,171 | PASS | 23,599 |
| tp_0001 | PASS | 11,031 | PASS | 23,040 | PASS | 32,688 |
| tp_0002 | PASS | 11,270 | PASS | 73,296 | PASS | 30,941 |
| tp_0003 | PASS | 10,757 | PASS | 21,764 | PASS | 28,051 |
| tp_0004 | PASS | 10,901 | PASS | 44,994 | PASS | 55,119 |
| tp_0005 | PASS | 15,697 | PASS | 20,756 | PASS | 100,796 |
| tp_0006 | PASS | 14,990 | PASS | 46,402 | PASS | 67,350 |
| tp_0007 | PASS | 22,553 | PASS | 29,686 | PASS | 72,521 |
| tp_0008 | PASS | 11,618 | PASS | 29,231 | PASS | 62,775 |
| tp_0009 | PASS | 13,229 | PASS | 42,912 | PASS | 86,005 |
| tp_0015 | PASS | 10,556 | FAIL | 25,337 | FAIL | 32,417 |
| tp_0016 | PASS | 10,336 | PASS | 20,363 | PASS | 25,948 |
| tp_0017 | PASS | 13,965 | PASS | 26,471 | PASS | 33,043 |
| tp_0018 | PASS | 11,017 | PASS | 33,277 | PASS | 51,395 |
| tp_0019 | PASS | 11,001 | PASS | 23,393 | FAIL | 31,988 |

### Medium (15 tasks)

| Task | OpenSymbolicAI | | LangChain | | CrewAI | |
|------|:-:|--:|:-:|--:|:-:|--:|
| | Pass | Tokens | Pass | Tokens | Pass | Tokens |
| tp_0010 | PASS | 13,955 | PASS | 87,130 | FAIL | 145,609 |
| tp_0011 | PASS | 14,285 | FAIL | 7,228 | FAIL | 128,475 |
| tp_0012 | PASS | 17,237 | PASS | 77,974 | PASS | 30,888 |
| tp_0013 | PASS | 12,952 | PASS | 93,953 | FAIL | 182,918 |
| tp_0014 | PASS | 19,879 | PASS | 84,884 | PASS | 132,085 |
| tp_0020 | PASS | 13,191 | PASS | 58,901 | PASS | 76,870 |
| tp_0021 | PASS | 12,848 | PASS | 56,648 | PASS | 81,541 |
| tp_0022 | PASS | 13,110 | PASS | 46,747 | PASS | 19,406 |
| tp_0023 | PASS | 12,233 | PASS | 58,227 | PASS | 84,193 |
| tp_0024 | PASS | 12,891 | PASS | 51,187 | PASS | 75,952 |
| tp_0030 | PASS | 10,798 | PASS | 27,601 | PASS | 33,956 |
| tp_0031 | PASS | 10,473 | PASS | 28,737 | PASS | 33,413 |
| tp_0032 | PASS | 10,566 | FAIL | 25,375 | FAIL | 39,653 |
| tp_0033 | PASS | 11,192 | FAIL | 28,050 | FAIL | 32,731 |
| tp_0034 | PASS | 10,659 | PASS | 34,824 | PASS | 41,873 |

### Hard (15 tasks)

| Task | OpenSymbolicAI | | LangChain | | CrewAI | |
|------|:-:|--:|:-:|--:|:-:|--:|
| | Pass | Tokens | Pass | Tokens | Pass | Tokens |
| tp_0025 | PASS | 12,317 | FAIL | 7,269 | PASS | 119,394 |
| tp_0026 | PASS | 16,106 | FAIL | 12,880 | PASS | 147,238 |
| tp_0027 | PASS | 12,518 | FAIL | 87,568 | FAIL | 136,402 |
| tp_0028 | PASS | 20,535 | PASS | 115,090 | FAIL | 141,246 |
| tp_0029 | PASS | 12,160 | PASS | 19,515 | PASS | 152,653 |
| tp_0035 | PASS | 13,417 | FAIL | 62,640 | FAIL | 88,334 |
| tp_0036 | PASS | 17,071 | PASS | 60,685 | PASS | 74,639 |
| tp_0037 | PASS | 12,948 | PASS | 12,807 | PASS | 82,333 |
| tp_0038 | PASS | 12,157 | PASS | 48,390 | PASS | 69,035 |
| tp_0039 | PASS | 12,356 | FAIL | 13,100 | FAIL | 70,912 |
| tp_0040 | PASS | 12,849 | FAIL | 96,292 | FAIL | 132,997 |
| tp_0041 | PASS | 43,321 | PASS | 139,361 | PASS | 193,667 |
| tp_0042 | PASS | 13,305 | PASS | 16,217 | FAIL | 141,517 |
| tp_0043 | PASS | 17,194 | PASS | 13,088 | PASS | 113,095 |
| tp_0044 | PASS | 12,882 | PASS | 17,576 | PASS | 122,246 |

</details>
