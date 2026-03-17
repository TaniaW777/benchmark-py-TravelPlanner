<h1  align="center">TravelPlanner Benchmark</h1>  <p  align="center">
<img  src="https://img.shields.io/badge/python-3.12+-blue.svg"  alt="Python Version">
<img  src="https://img.shields.io/badge/License-MIT-yellow.svg"  alt="License">
</p>
  
## Overview

TravelPlanner is a rigorous evaluation framework designed to measure the strategic planning capabilities of Large Language Models (LLMs ). Unlike simple QA benchmarks, it requires agents to navigate complex, multi-day itineraries while satisfying a variety of real-world constraints:

-  **Goal-Oriented Reasoning:** Managing budgets, transportation, and accommodation simultaneously.

-  **Tool Interaction:** Efficiently querying flights, hotels, and local databases.

-  **Constraint Satisfaction:** Respecting specific user preferences (e.g., "no flights" or "halal cuisine") and commonsense logic.

This repository implements the benchmark using the **GoalSeeking** pattern, providing a standardized environment to track performance, token efficiency, and execution costs across different LLM architectures.
  

## Quickstart 

Run your first benchmark in under 5 minutes.
  
**1. Clone the repository and install dependencies:**

   #### a-Clone the repository
  

```bash
git  clone  https://github.com/OpenSymbolicAI/benchmark-py-TravelPlanner.git
```
```bash
cd  benchmark-py-TravelPlanner
```
 #### b-Install dependencies
```bash
uv  sync
```  

**2. Configure API Keys:**

Create a `.env` file in the project root. Add the API key for your desired LLM provider.
```

# LLM provider (pick one based on --provider flag )

FIREWORKS_API_KEY=... # Default provider - https://fireworks.ai

OPENAI_API_KEY=sk-... # For --provider openai

ANTHROPIC_API_KEY=sk-ant-... # For --provider anthropic

GROQ_API_KEY=gsk_... # For --provider groq

```
 

**3. Run a small subset (5 tasks)**

```bash
uv  run  python  -m  travelplanner_bench.runner  --split  train  --num  5
```  

For detailed usage and commands, see the [Usage](#usage) section below.

 
## Results Summary
 
Current baseline results for supported providers:
| Model | Pass Rate | Avg Tokens | Avg Cost | Date |
|:------|:---------:|:----------:|:--------:|:----:|
| | | | | |
| | | | | |

### Framework Comparison at a Glance

Same model, same tools, same evaluation — only the framework differs. Full analysis in [**COMPARISON.md**](COMPARISON.md). For multi-model results across 11 LLMs and 4 providers, see [**MODEL-LANDSCAPE.md**](MODEL-LANDSCAPE.md).


| | OpenSymbolicAI | LangChain | CrewAI |
| --- | --- | --- | --- |
| **Pass Rate** | **100%** | 77.8% | 73.3% |
| **Tokens / Task** | **13,936** | 43,801 (3.1x) | 81,331 (5.8x) |
| **LLM Calls / Task** | **2.3** | 13.5 (5.9x) | 39.6 (17x) |
| **Cost / Passing Task** | **$0.013** | $0.051 (4.1x) | $0.100 (8x) |
| **Latency** | **47s** | 73s (1.5x) | 124s (2.6x) |
  

> Multipliers show how much more each framework consumes relative to OpenSymbolicAI. Measured on 45 train tasks (15 easy + 15 medium + 15 hard) with `gpt-oss-120b` via Fireworks AI.  

## Detailed Results

### Train Split (45 tasks)  

| Level | Tasks | Delivery | Commonsense | Hard Constraints | Final Pass | Avg Time |
| --- | --- | --- | --- | --- | --- | --- |
| Easy | 15 | 100% | 100% | 100% | **100%** | — |
| Medium | 15 | 100% | 100% | 100% | **100%** | — |
| Hard | 15 | 100% | 100% | 100% | **100%** | — |
| **All** | **45** | **100%** | **100%** | **100%** | **100%** | **52.6s** |
 

### Validation Split (180 tasks)

| Level | Tasks | Delivery | Commonsense | Hard Constraints | Final Pass | Avg Time |
| --- | --- | --- | --- | --- | --- | --- |
| Easy | 60 | 100% | 100% | 100% | **100%** | — |
| Medium | 60 | 100% | 98.3% | 100% | **98.3%** | — |
| Hard | 60 | 100% | 100% | 100% | **100%** | — |
| **All** | **180** | **100%** | **99.4%** | **100%** | **99.4%** | **55.5s** |
  

### Test Split (1,000 tasks)  

| Level | Tasks | Delivery | Commonsense | Hard Constraints | Final Pass | Avg Time |
| --- | --- | --- | --- | --- | --- | --- |
| Easy | 348 | 100% | 98.6% | 100% | **98.6%** | — |
| Medium | 333 | 100% | 97.0% | 100% | **97.0%** | — |
| Hard | 319 | 100% | 98.1% | 100% | **98.1%** | — |
| **All** | **1,000** | **100%** | **97.9%** | **100%** | **97.9%** | **52.4s** |

  

### Comparison with Published Baselines

Results from the [TravelPlanner paper](https://arxiv.org/abs/2402.01622) (ICML 2024) on the validation split:
 

| Method | Delivery | Commonsense | Hard | Final Pass |
| --- | --- | --- | --- | --- |
| GPT-3.5-Turbo | 100% | 2.9% | 1.7% | 0.6% |
| GPT-4 | 100% | 6.4% | 3.7% | 0.6% |
| GPT-4-Turbo | 99.4% | 11.7% | 4.6% | 4.4% |
| Gemini 1.5 Pro | 98.3% | 7.8% | 4.5% | 3.9% |
| **OpenSymbolicAI (ours)** | **100%** | **99.4%** | **100%** | **99.4%** |

  

## What is TravelPlanner?


TravelPlanner gives the agent a natural language travel request and asks it to produce a complete itinerary:

  
>  **Query:** Plan a 3-day trip from Sarasota to Chicago for 1 person with a budget of $1,900, from March 22nd to March 24th, 2022.
  

The agent must:

1.  **Search** for flights, restaurants, accommodations, and attractions

2.  **Plan** a day-by-day itinerary with transportation, meals, sightseeing, and lodging

3.  **Satisfy** all explicit constraints (budget, cuisine, room type, etc.)  

5.  **Respect** commonsense rules (no duplicate restaurants, valid city routes, etc.)

  

### Difficulty Levels

| Level | Description | Constraints |
| --- | --- | --- |
| **Easy** | Single city, 1 person | Budget only |
| **Medium** | Single/multi city, 2-8 people | Budget + 1 constraint (cuisine, room type, or room rule) |
| **Hard** | Multi-city, variable group | Budget + 3 constraints (cuisine + room type/rule + transportation) |

  

### Dataset

| Split | Size | Purpose |
| --- | --- | --- |
| Train | 45 | Human-annotated reference plans |
| Validation | 180 | Evaluation with ground truth |
| Test | 1,000 | Blind test set |

  

Source: [HuggingFace `osunlp/TravelPlanner`](https://huggingface.co/datasets/osunlp/TravelPlanner)

  

## Evaluation Metrics

The benchmark evaluates four categories with 13 individual checks:

### Commonsense Constraints (8 checks)

  

| Check | Description |
| --- | --- |
| Within Sandbox | All entities (flights, restaurants, hotels, attractions) exist in the database |
| Complete Information | No excessive missing fields in the itinerary |
| Within Current City | Daily activities match the designated city |
| Reasonable City Route | Starts from origin, returns to origin, logical sequence |
| Diverse Restaurants | No restaurant visited more than once |
| Diverse Attractions | No attraction visited more than once |
| Non-Conflicting Transport | No mixing self-driving with flights |
| Valid Accommodation | Meets minimum-nights requirements |

  

### Hard Constraints (5 checks)

  

| Check | Description |
| --- | --- |
| Budget | Total cost (flights + meals + accommodation) within stated budget |
| Room Rule | Accommodation complies with house rules (no smoking, no parties, etc.) |
| Room Type | Correct room type (entire home, private room, shared room) |
| Cuisine | All required cuisines represented in meals |
| Transportation | Forbidden transport mode not used (e.g., "no flights") |

  

## Usage

  

### Run by Difficulty

  

```bash

# Easy tasks only (budget constraint, single city)

uv  run  python  -m  travelplanner_bench.runner  --model  gpt-4o  --provider  openai  --level  easy

  

# Medium tasks (budget + 1 constraint)

uv  run  python  -m  travelplanner_bench.runner  --model  gpt-4o  --provider  openai  --level  medium

  

# Hard tasks (budget + 3 constraints, multi-city)

uv  run  python  -m  travelplanner_bench.runner  --model  gpt-4o  --provider  openai  --level  hard

```

  

### Full Validation Set (180 tasks)

  

```bash
uv  run  python  -m  travelplanner_bench.runner  --model  gpt-4o  --provider  openai  --parallel  5
```

  

### More Models

  



#### Anthropic Claude
```bash
uv  run  python  -m  travelplanner_bench.runner  --model  claude-sonnet-4-20250514  --provider  anthropic  --num  10
```
  

#### Fireworks (open-source models)
```bash
uv  run  python  -m  travelplanner_bench.runner  --model  gpt-oss-120b  --provider  fireworks  --num  10
```
  

#### Local Ollama
```bash
uv  run  python  -m  travelplanner_bench.runner  --model  llama3  --provider  ollama  --num  5
```


### Observability
  
Send structured traces to the local observability stack for per-span inspection of planning, execution, and goal-seeking iterations:

#### Start the observability stack (collector + dashboard)
```bash
cd  /path/to/OpenSymbolicAI/observability && docker  compose  up
```
  
#### Run with tracing enabled
```bash
uv  run  python  -m  travelplanner_bench.runner  --model  gpt-oss-120b  --provider  fireworks  --num  5  --observe
```

### CLI Reference

For a full list of commands and options:


```bash
uv  run  python  -m  travelplanner_bench.runner  --help
```

## Contributing

We welcome contributions! Please ensure your pull requests follow standard branch hygiene:

  
  
1. Fork the repo and create a feature branch.
  

1. Ensure all commands in the README are verified locally.

  
1. Open a Pull Request with a clear description of changes.