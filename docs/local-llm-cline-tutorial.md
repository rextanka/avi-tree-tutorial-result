# Local LLM Development with Cline + Ollama on Apple Silicon

> **Status: Draft, for review and refinement**

---

## Introduction

This tutorial sets up a fully local, private AI-assisted coding workflow on an Apple Silicon Mac. Everything runs on-device: no API keys, no cloud calls, and no data leaving the machine.

The stack is:

| Layer | Tool |
|---|---|
| Agent | Cline (VS Code extension) |
| Model server | Ollama 0.19+ |
| Model | `gemma3-cline` (24GB) or `qwen35b-cline` (48GB) — local Ollama variants |
| Language | C++20 |
| Build system | CMake 3.24+ |
| Unit tests | GoogleTest via FetchContent |
| Functional tests | pytest + subprocess |

The tutorial walks through installation and configuration, then uses a worked example (building an AVL tree in C++) to demonstrate the three-phase Plan/Act workflow: infrastructure, interface and unit tests, implementation and functional tests.

The guidance is calibrated for two reference machines: a **24GB MacBook Air (M4)** as the baseline tight-budget configuration, and a **48GB MacBook Pro (M4 Pro)** as a higher-tier option. Both run macOS Tahoe 26.5 with Ollama 0.19+ and use the same workflow. The divergence points (model choice, context window, expected timing) are flagged inline. On 16GB hardware some operations will be tighter and `num_ctx` will need to drop to 8192. Larger machines (64GB+, Mac Studio, Mac Pro) can run larger models still; the principles below scale.

**On time-sensitive content.** Most of this tutorial (Cline workflow, git pattern, CMake setup, sub-prompt structure) is stable. Model-specific recommendations are not: new models ship monthly, quantisation formats evolve, and Ollama's backend changes. Sections that name specific models or quote throughput numbers are explicitly marked with a "Last reviewed" date. Plan to revisit those sections every three months or so.

For background on MLX, quantisation, the two test layers, and why an AVL tree was chosen for the example, see **Appendix A: Background** at the end of this document. The setup steps below do not require reading the appendix first.

---

## Prerequisites

Install in this order. Later items depend on earlier ones.

**Homebrew.** The package manager used for everything else. If not installed:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Or see [brew.sh](https://brew.sh) for the canonical instructions.

**Xcode Command Line Tools** (provides `clang++`, `git`, `make`):

```bash
xcode-select --install
```

If already installed, this prints "already installed". Safe to run regardless.

**CMake:**

```bash
brew install cmake
# Should report 3.24 or later:
cmake --version
```

**Python 3.10 or later.** Required for the pytest functional test layer.

```bash
python3 --version
```

If 3.9 or below:

```bash
brew install python@3.12
```

Homebrew will print PATH instructions after installation. Follow them, open a new terminal, and confirm:

```bash
# Should report 3.12.x:
python3 --version
```

If it still shows the old version, use `python3.12` explicitly throughout.

**VS Code.** Install from [code.visualstudio.com](https://code.visualstudio.com).

**VS Code `code` CLI.** Open VS Code, press `Cmd+Shift+P`, and run **Shell Command: Install 'code' command in PATH**. Confirm with:

```bash
# Should print a three-line version string (VS Code version, commit, arch):
code --version
```

If you get "command not found", repeat the install step inside VS Code.

---

## Step 1: Install Ollama

```bash
brew install ollama
```

If you have an existing Ollama install, update it to ensure the MLX backend is active:

```bash
brew upgrade ollama
# Should report 0.19.0 or later:
ollama --version
```

**Choose one startup method. Do not use both:**

| Method | Command | When to use |
|---|---|---|
| Foreground | `ollama serve` | Testing; logs visible in terminal |
| Background service | `brew services start ollama` | Day-to-day; starts on login |

Running both causes a port conflict on `11434`.

> **Security note:** Ollama binds to `localhost` by default. Do not change this to `0.0.0.0`, since Ollama has no authentication layer.

---

## Step 2: Pull a Model and Configure It

> **Last reviewed: June 2026.** The specific models named below reflect what was current and recommended at that time. The Cline + Ollama + Modelfile workflow itself is stable; the model names are not. Check [Ollama's library](https://ollama.com/library), [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA), and the "Finding Models to Experiment With" section near the end of this document for current alternatives.

### Choose a model for your hardware tier

Three models are recommended, organised by what runs comfortably on each machine tier.

**For 24GB (MacBook Air M4 baseline):**

| Model | Pull command | Size (4-bit) | Context window | Notes |
|---|---|---|---|---|
| Gemma 3 12B | `ollama pull gemma3:12b` | ~8 GB | 128K tokens | Stronger reasoning; good for complex tasks |
| Qwen 3.5 9B | `ollama pull qwen3.5:9b` | ~6.6 GB | 256K tokens | Faster; larger context suits longer sessions |

**For 48GB (MacBook Pro M4 Pro and above):**

| Model | Pull command | Size | Context window | Notes |
|---|---|---|---|---|
| Qwen 3.5 35B-A3B (NVFP4, coding) | `ollama pull qwen3.5:35b-a3b-coding-nvfp4` | ~22 GB | 256K tokens | Mixture-of-Experts: 35B total params, only 3B active per token. NVFP4 quantisation. Tuned for coding. |
| Gemma 3 12B | `ollama pull gemma3:12b` | ~8 GB | 128K tokens | Still useful for quick interactive work; loads faster than the 35B and leaves more memory for other applications |
| Qwen 3.5 9B | `ollama pull qwen3.5:9b` | ~6.6 GB | 256K tokens | Fastest option when capability is not the limit |

On a 48GB machine, the Qwen 3.5 35B-A3B is the headline option: substantially more capable than the smaller dense models, and the MoE architecture keeps per-token decode roughly comparable to a dense 3B model despite the larger total parameter count. NVFP4 quantisation is a newer format ([introduced by NVIDIA](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)) that maintains accuracy better than Q4_K_M at similar memory cost; Ollama 0.19 added native support and tuned this model variant specifically for coding workloads.

Pulling the smaller models on a 48GB machine is still worthwhile. For exploratory chat, quick edits, or interactive prompts where snappy turnaround matters more than the strongest reasoning, the 9B or 12B model will respond faster (smaller model = less time loading per cold start, less memory pressure on the rest of the system). Choose per task. Cline's model setting is one configuration change away.

> Ollama recommends "more than 32GB of unified memory" as the minimum for the 35B-A3B model. The 24GB Air cannot run it.

For this tutorial, pick one model from the relevant tier and use it throughout. The walkthrough uses `gemma3:12b` for the 24GB case and `qwen3.5:35b-a3b-coding-nvfp4` as the recommended upgrade path on 48GB. Pull your chosen model:

```bash
# 24GB tier (or for use alongside the 35B on 48GB):
ollama pull gemma3:12b

# 48GB headline:
ollama pull qwen3.5:35b-a3b-coding-nvfp4
```

Note: Ollama's MLX architecture support is expanding with each release. If generation on a new model is unexpectedly slow, check the [Ollama GitHub issues](https://github.com/ollama/ollama) to confirm MLX is active for that architecture.

### Create a Persistent Modelfile

Save to `~/ollama-models/` so it can be recreated after any Ollama reinstall:

```bash
mkdir -p ~/ollama-models
cat > ~/ollama-models/gemma3-cline.modelfile << 'EOF'
FROM gemma3:12b

# 16K context is a practical balance for 24GB.
# Reduce to 8192 if Activity Monitor shows persistent memory pressure.
PARAMETER num_ctx 16384

# Lower temperature produces more deterministic output, preferable for code generation.
PARAMETER temperature 0.2

# Penalises repetition. Reduces looping behaviour in smaller models.
PARAMETER repeat_penalty 1.1
EOF

ollama create gemma3-cline -f ~/ollama-models/gemma3-cline.modelfile
# gemma3-cline should appear in the listing:
ollama list
```

For Qwen 3.5 9B, use the equivalent block:

```bash
cat > ~/ollama-models/qwen-cline.modelfile << 'EOF'
FROM qwen3.5:9b

# 16K context is a practical balance for 24GB.
# Reduce to 8192 if Activity Monitor shows persistent memory pressure.
PARAMETER num_ctx 16384

# Lower temperature produces more deterministic output, preferable for code generation.
PARAMETER temperature 0.2

# Penalises repetition. Reduces looping behaviour in smaller models.
PARAMETER repeat_penalty 1.1
EOF

ollama create qwen-cline -f ~/ollama-models/qwen-cline.modelfile
# qwen-cline should appear in the listing:
ollama list
```

For the **48GB tier headline model**, use a larger context window since memory headroom permits it:

```bash
cat > ~/ollama-models/qwen35b-cline.modelfile << 'EOF'
FROM qwen3.5:35b-a3b-coding-nvfp4

# 32K context is reasonable on 48GB. The model supports up to 256K, but
# context KV cache scales linearly with size and eats into headroom.
# 32K covers all typical agentic coding sessions comfortably.
PARAMETER num_ctx 32768

# Lower temperature produces more deterministic output, preferable for code generation.
# This model variant ships with sampling parameters already tuned for coding,
# but explicit values keep behaviour reproducible.
PARAMETER temperature 0.2

# Penalises repetition. Still useful even on larger models.
PARAMETER repeat_penalty 1.1
EOF

ollama create qwen35b-cline -f ~/ollama-models/qwen35b-cline.modelfile
ollama list
```

> **Context window note for 48GB users.** You have memory budget to push `num_ctx` higher than the 24GB recommendation. 32K is a comfortable default. 64K is achievable for the 35B-A3B model (the MoE architecture means lower per-token memory cost than a dense 35B would imply). Going above 64K is possible but consumes memory you may want for other applications. Match Cline's Context Window setting (Step 4) to whatever you choose.

### Keep the Model Loaded Between Steps

By default, Ollama unloads a model from memory after 5 minutes of inactivity. The next request triggers a cold reload that adds 30 to 90 seconds of lag. To keep the model resident, set the `OLLAMA_KEEP_ALIVE` environment variable so Ollama reads it at startup. `-1` keeps the model loaded indefinitely; other valid values are duration strings such as `30m` or `24h`, or an integer number of seconds.

**The reliable approach: run Ollama in foreground with the variable inline.**

In a dedicated terminal tab, stop the brew service and run:

```bash
brew services stop ollama
OLLAMA_KEEP_ALIVE=-1 ollama serve
```

Leave the tab open while working. Logs are visible if anything goes wrong, and the variable is guaranteed to be in effect because the process inherits it directly from your shell. Press Ctrl+C when done; restart the brew service with `brew services start ollama` if you want background-managed Ollama for other tools.

**Verify the variable was picked up.** Ollama's startup log includes a `server config` line listing every environment setting in effect. Look for `OLLAMA_KEEP_ALIVE:` followed by an unusually large duration such as `2562047h47m16.854775807s`. That ridiculous-looking value is Go's `time.Duration` maximum, which is how `-1` is represented internally. Seeing it confirms the variable was set; this is correct behaviour, not an error.

After loading a model (the first Cline request triggers this), confirm via `ollama ps` in a separate terminal tab. The UNTIL column should show "Forever" rather than a countdown.

**Hiding log output.** Ollama's foreground process prints log lines continuously into your terminal, which can be disruptive if you also want to use that tab for other commands. Redirect to a file instead:

```bash
OLLAMA_KEEP_ALIVE=-1 ollama serve > /tmp/ollama.log 2>&1 &
```

The `&` backgrounds the process within the current shell (it still dies when the tab closes). Watch logs on demand with `tail -f /tmp/ollama.log` in another tab.

**While foreground Ollama is running, `brew services` will correctly report it as not running.** Output like the following is expected and does not indicate a problem:

```
$ brew services info ollama
ollama (homebrew.mxcl.ollama)
Running: ✘
Loaded: ✘
```

The brew-managed service is genuinely stopped; your foreground process is the one serving port 11434. Do not attempt to start the brew service while foreground Ollama is running; it will fail with a port conflict.

**A note on Ollama's default context.** When Ollama starts, it logs a line like `vram-based default context" total_vram="17.8 GiB" default_num_ctx=4096`. This is the *global* default for models that do not specify their own context window. Your `gemma3-cline` Modelfile sets `PARAMETER num_ctx 16384`, which takes precedence; you do not need to change this default.

**Why not the brew background service?** The brew-managed launchd job does not read your shell startup files, so setting `OLLAMA_KEEP_ALIVE` in `~/.zshrc` has no effect on the service. Editing the underlying plist file is possible in principle, but its location varies between Homebrew versions (some installs have no plist file in `~/Library/LaunchAgents/` at all) and `brew services` may regenerate it on certain operations. The foreground approach sidesteps all of this.

**Alternative: accept the 5-minute default.** A 30 to 90 second cold reload once every 5 minutes of idle is not catastrophic. The keep-alive setup matters most for active, multi-step Cline sessions where reloads would interrupt flow. For exploratory use, the default is fine and the brew service can be left running.

Memory note: keeping a model loaded indefinitely occupies its share of unified memory continuously. On a 24GB machine running an 8GB model, that leaves around 10 GB for other applications. On a 48GB machine running the 22GB Qwen 3.5 35B-A3B model, that leaves around 22 GB for everything else — comfortable headroom. If you also run memory-intensive workloads, prefer a finite value such as `30m`.

---

## Step 3: Install Cline in VS Code

1. Open VS Code.
2. Open the Extensions panel (`Cmd+Shift+X`).
3. Search for **Cline**.
4. Install the extension by **saoudrizwan** (ID: `saoudrizwan.claude-dev`). Do not install forks or similarly named extensions.

---

## Step 4: Configure Cline to Use Ollama

1. Open the Cline panel (sidebar icon, or `Cmd+Shift+P` then "Cline: Open Panel").
2. Click the **Settings** gear icon.
3. Set **API Provider** to `Ollama`.
4. Set **Base URL** to `http://localhost:11434`.
5. Leave **Ollama API Key** empty. Local Ollama has no authentication; anything in this field will cause "Invalid URL" errors.
6. Set **Model** to your chosen Modelfile variant: `gemma3-cline` (24GB), `qwen-cline` (24GB alternative), or `qwen35b-cline` (48GB).
7. Set Cline's **Context Window** to match the `num_ctx` value in your Modelfile: `16384` for the 24GB models, `32768` for `qwen35b-cline`. This setting is independent of the Modelfile value; both must match or you will not get the full context budget.
8. Set **Request Timeout (ms)** to `300000` (5 minutes). The default of 30 seconds is too short for the first request after a model loads from cold, which can take 30 to 90 seconds on a 24GB machine. Once the model is resident in memory, responses are fast and the timeout only matters for cold starts.
9. **Auto-approve settings.** Cline lets you pre-authorise certain tool categories (Read, Edit, Safe Commands, MCP) so the model can act without prompting for each one. Frontier models can be trusted with reasonable auto-approve sets; smaller local models cannot. A 9-12B model will occasionally propose wrong file writes (creating files where directories should be), wrong terminal commands (`cmake .` instead of `cmake -B build`), or hallucinated paths. Broad auto-approve amplifies the damage these mistakes can do before you notice.
   
   For this tutorial, set auto-approve to **Read only** (or disable it entirely). This requires you to click Approve on each Edit and command, which is the entire point of the human-in-the-loop workflow. You can revisit this later if you move to a stronger model or trust a specific project.
10. Save settings.

---

## Step 5: Bootstrap the Project

This step is done **manually**, before running any Cline prompts. Cline reads `.clinerules` from the project root at the start of every task, so the file must exist before Cline runs.

```bash
mkdir avl-tree-cpp && cd avl-tree-cpp
git init
```

Create the venv for the pytest functional test layer:

```bash
python3 -m venv .venv
source .venv/bin/activate
.venv/bin/pip install pytest
```

Create `.clinerules`:

```bash
cat > .clinerules << 'EOF'
You are running locally on a resource-constrained environment (Apple Silicon, 24GB unified memory).

Project: C++20, CMake 3.24+, GoogleTest via FetchContent.
Build directory: build/. Always invoke cmake and build commands from the project root.
Functional tests: Python/pytest in tests/functional/. Use .venv/bin/python3 and
.venv/bin/pytest; never use the system python.

- Read only the specific file or function named in the task. Do not scan entire directories.
- Write tight, focused diffs. Do not rewrite files when a small change will do.
- Complete one discrete change per step and stop for confirmation.
- Do not run directory-wide search or analysis unless explicitly asked.
- Use `cmake --build build/` to build. Use `ctest --test-dir build/ --output-on-failure`
  to run unit tests.
- All new C++ headers must use `#pragma once`. All code must target C++20.
- Plan mode is read-only. In Plan mode, describe the steps you would take but
  do not invoke write_to_file, execute_command, or any other tool that modifies
  files or runs commands. Use plan_mode_respond to present your plan and wait
  for the user to switch to Act mode before executing.
- To create directories, use execute_command with `mkdir -p <path>`. Do not
  attempt to create a directory using write_to_file; write_to_file only creates
  files. If a task requires both directories and files, always run mkdir first,
  then create files inside the new directories.
EOF
```

Add `.gitignore` entries for the venv and build artefacts:

```bash
cat > .gitignore << 'EOF'
build/
.cache/
compile_commands.json
.DS_Store
*.dSYM
.venv/
EOF
```

Open the project in VS Code:

```bash
code .
```

Confirm the venv is active in VS Code's integrated terminal (look for `(.venv)` in the prompt). If absent, run `source .venv/bin/activate` or press `Cmd+Shift+P`, choose "Python: Select Interpreter", and select `.venv/bin/python3`.

The `.clinerules` file is intentionally short and concrete. See the **Reference: `.clinerules` and Optimisation** section for the principles behind it. If you adapt it for a different project or language, remove or update the Python venv line; pointing Cline at a `.venv` that does not exist will cause spurious errors.

### Commit the bootstrap

Before letting Cline make any changes, commit the bootstrap state to git. This gives you a known-good baseline to return to if something goes wrong later.

```bash
git add .
git status      # confirm .venv/ and build/ are excluded
git commit -m "Bootstrap: gitignore, clinerules, and venv setup"
```

**Why this matters.** Subsequent steps will hand Cline broad permission to create files, modify existing ones, and run commands. Local models occasionally produce wrong output: an over-eager refactor that breaks a working file, an `rm` in a terminal step you approved too quickly, or a misinterpretation that rewrites the wrong file. Committing at clean milestones means any of these failures becomes a one-command recovery (`git restore`, `git reset`, or `git checkout`) rather than a debugging session.

The pattern this tutorial uses: commit after each successful Cline step (Step 6, 7, 8, and so on). If a step goes sideways and you cannot easily correct the agent's output, `git reset --hard HEAD` restores the previous good state and you can retry the prompt with adjusted wording. Cline's own Checkpoints feature provides a similar safety net at the level of individual Act steps (see Step 10), but git commits are more durable, survive across sessions, and travel with the repository.

**If you are new to git**, the [Pro Git book](https://git-scm.com/book/en/v2) is the canonical free reference; the first two chapters cover everything you need for this tutorial. For a quicker on-ramp, the [GitHub git handbook](https://docs.github.com/en/get-started/using-git/about-git) is more concise. The commands you will use most in this tutorial are:

- `git status` — what has changed since the last commit
- `git diff` — show the actual changes
- `git add .` — stage all current changes
- `git commit -m "message"` — save the staged changes as a commit
- `git log --oneline` — list previous commits
- `git restore <file>` — undo unstaged changes to a file
- `git reset --hard HEAD` — discard all uncommitted changes and return to the last commit (destructive; use with care)

---

## Step 6: Infrastructure

The next three steps each consist of several small, focused sub-prompts rather than one large prompt. This subdivision is deliberate. Local 9B-12B models behave better when each prompt has a single concrete deliverable and a single verifiable outcome. Cline's own system prompt, tool definitions, your `.clinerules`, and any files Cline reads while planning all consume context tokens before your prompt even arrives, leaving less room than you might expect inside a 16K window. Smaller prompts also produce smaller diffs, which are easier to review and correct.

Each sub-prompt is followed by a verification step and a commit. The git pattern is: branch at the start of the step, commit after each sub-prompt, squash-merge the branch back to main at the end. If any sub-prompt produces wrong output you cannot easily correct, `git reset --hard HEAD~1` discards that sub-prompt and you retry. If the whole step goes sideways, `git checkout main && git branch -D step-6-infra` discards everything.

### Expectations on time and effort

Running these sub-prompts on a **24GB Apple Silicon machine with `gemma3:12b`** typically takes 2 to 5 minutes per sub-prompt of model generation time, plus your review and approval time per tool call. A complete step (4-5 sub-prompts) commonly takes 20 to 40 minutes elapsed.

On a **48GB MacBook Pro with `qwen3.5:35b-a3b-coding-nvfp4`**, per-sub-prompt timing is broadly similar in elapsed terms (1 to 4 minutes), despite the model being far larger. The MoE architecture activates only 3B parameters per token, so decode is much faster than the parameter count suggests. Prefill (processing your prompt and Cline's system prompt) is where the larger model is meaningfully faster, because MLX on Apple Silicon prefills aggressively. The active cooling on the Pro chassis also lets the M4 Pro sustain higher throughput than the fanless Air under load.

This is the cost of running local; it buys privacy, no API spend, and full control over your workflow, but it is genuinely slower than calling a frontier API. Plan your sessions accordingly: a single step is roughly one focused work block.

Plan-mode failure loops (described below) can blow this budget out by an additional 5 to 15 minutes if they trigger. Following the Act-mode recommendation reduces but does not eliminate the chance of these.

### On Plan vs Act mode for smaller models

Cline supports two modes: Plan mode (read-only, no tool execution) and Act mode (can write files and run commands). Frontier models respect this distinction natively. Smaller local models (9B-12B) often do not.

During testing with `gemma3:12b`, sub-prompts run in Plan mode would frequently trigger a failure loop. The model would attempt to call `write_to_file` despite being in Plan mode. Cline would correctly refuse the tool call with an error message. The model would not understand the refusal and would retry the same tool call dozens of times. Two outcomes are possible:

- The model eventually self-corrects, reformulates as a proper Plan-mode response, and presents its plan. Typically 60 to 90 seconds wasted.
- The loop runs long enough that Cline's automatic context pruning kicks in, silently discarding earlier conversation turns to keep the context window from overflowing. This often destroys the model's previous good output (the plan it had already drafted), leaving an empty re-presented plan when the model finally recovers. There is no recovery from this except resubmitting from scratch.

The `.clinerules` file (Step 5) includes a rule telling the model that Plan mode is read-only. This helps but does not reliably prevent the loop on smaller models.

**Recommendation: run every sub-prompt in Act mode.** Cline's per-tool-approval in Act mode is itself a form of plan/act gating; you see each proposed tool call (file write, shell command) and approve or reject before it executes. This is functionally equivalent to Plan-mode review without the loop risk. The trade-off is more clicks per sub-prompt; the gain is far fewer recovery cycles. Frontier-model users can use Plan mode safely, but for the local-9B-12B path this tutorial assumes, Act mode is the safer default.

Per-sub-prompt mode recommendations below all reflect this. If you find Plan mode works on your specific model, override as you see fit.

### On prompt specificity

A small model takes vague prompts as licence to improvise, and its improvisations are often subtly wrong in ways that cost recovery cycles. The sub-prompts below are specific by design. Resist the urge to summarise them or paraphrase them when typing into Cline; the verbosity has been tuned during testing.

For your own prompts later (in Step 10 or future projects), the same principle applies. Concrete examples from this tutorial's testing:

| Vague (causes problems) | Specific (works reliably) |
|---|---|
| "build the project" | "Build only the `avl_tree_unit_tests` target: `cmake --build build/ --target avl_tree_unit_tests`" |
| "fix the test" | "The test `BalancedAfterSortedInsert` expects `height() <= 4` but is currently returning 6. The `_rebalance` function is not triggering the Left-Right rotation case correctly. Fix only `_rebalance`." |
| "add a delete method" | "Add `bool remove(const std::string& key)` to the public interface in `avl_tree.h`. It should remove the node, rebalance, and return true. Return false if the key is not present. Do not modify any existing methods." |

The vague form invites the model to interpret intent broadly, scan files outside the task scope, propose unrelated changes, or run commands that have unintended side effects. The specific form constrains the task to a single tractable scope with a single verifiable outcome.

### Branch for the step

```bash
git checkout -b step-6-infra
```

### Sub-prompt 6.1: Create the directory layout and placeholder files

*Recommended mode: Act mode. The task is mechanical; approve each tool invocation as it arrives.*

This sub-prompt creates directories first via a single shell command, then places empty files inside them. The separation matters: Cline's `write_to_file` tool creates files only — not directories — and smaller models sometimes try to call `write_to_file` on a directory path, which creates a file at that path and breaks subsequent file creation inside it. The explicit two-phase structure below avoids this.

> **Note: this sub-prompt is a reasonable candidate to skip Cline entirely.** The work is purely mechanical (no judgement, no code, no syntax decisions) and can be done from your terminal in a few seconds with certainty:
>
> ```bash
> mkdir -p include/avl_tree src tests/unit tests/functional app
> touch include/avl_tree/avl_tree.h src/avl_tree.cpp tests/unit/avl_tree_test.cpp tests/functional/test_app.py app/main.cpp
> ```
>
> The point of using Cline is to learn the agentic workflow, not to prove the model can do trivial tasks. If you prefer to skip the prompt below, run those two commands, jump to the Verify step, and continue from there. The rest of the tutorial assumes the same end state regardless of how you got there.

If you do want to run it through Cline, use the prompt below.

```text
This task has two phases. Complete them in order.

Phase 1: Create the directory structure with a single mkdir command. Use
execute_command (not write_to_file) to run:

  mkdir -p include/avl_tree src tests/unit tests/functional app

Phase 2: Create five empty placeholder files (one comment line is acceptable
in each) using write_to_file. The directories from Phase 1 must already exist:

  include/avl_tree/avl_tree.h
  src/avl_tree.cpp
  tests/unit/avl_tree_test.cpp
  tests/functional/test_app.py
  app/main.cpp

Do not create any CMakeLists.txt files in this step. Do not modify .clinerules
or .gitignore; they already exist.
```

**Verify:**

```bash
# All five directories should be directories, not files:
file include/avl_tree src tests/unit tests/functional app
# All five files should exist and be empty or near-empty:
find include src tests app -type f
```

If `file` reports any of the five paths as something other than "directory" (e.g. "empty" or "ASCII text"), Cline created a file where a directory should be — see Troubleshooting.

**Commit:**

```bash
git add .
git commit -m "Step 6.1: directory layout and placeholders"
```

### Sub-prompt 6.2: Write the root CMakeLists.txt

*Recommended mode: Act mode. Review each proposed write carefully before approving; CMake configuration is easy to get subtly wrong, so look closely at the diff Cline presents before clicking Approve.*

```text
Write the root CMakeLists.txt for the avl-tree project. Do not modify any
other files. The content must include:

- cmake_minimum_required(VERSION 3.24)
- project(avl-tree VERSION 0.1 LANGUAGES CXX)
- set(CMAKE_CXX_STANDARD 20)
- set(CMAKE_CXX_STANDARD_REQUIRED ON)
- set(CMAKE_CXX_EXTENSIONS OFF)
- include(CTest) at top level
- include(FetchContent) before any FetchContent_Declare call
- FetchContent block for GoogleTest, pinned to tag v1.14.0 exactly
  (use FetchContent_Declare and FetchContent_MakeAvailable)
- After FetchContent_MakeAvailable(googletest), call include(GoogleTest)
- add_subdirectory(src)
- add_subdirectory(tests/unit)
- add_subdirectory(app)

Do not add any other directives.
```

**Verify:** open `CMakeLists.txt` and confirm:
- the v1.14.0 tag is present (not `main`, not a different version)
- `include(FetchContent)` appears before `FetchContent_Declare`
- `include(GoogleTest)` appears after `FetchContent_MakeAvailable`
- `include(CTest)` is at top level before the `add_subdirectory` calls

**Commit:**

```bash
git add CMakeLists.txt
git commit -m "Step 6.2: root CMakeLists.txt"
```

### Sub-prompt 6.3: Write the subdirectory CMakeLists.txt files

*Recommended mode: Act mode. Review each proposed write before approving.*

```text
Write three CMakeLists.txt files. Do not modify any other files.

1. src/CMakeLists.txt
   - add_library(avl_tree_lib STATIC avl_tree.cpp)
   - target_include_directories(avl_tree_lib PUBLIC ${CMAKE_SOURCE_DIR}/include)

2. tests/unit/CMakeLists.txt
   - add_executable(avl_tree_unit_tests avl_tree_test.cpp)
   - target_link_libraries(avl_tree_unit_tests PRIVATE avl_tree_lib GTest::gtest_main)
   - gtest_discover_tests(avl_tree_unit_tests)

3. app/CMakeLists.txt
   - add_executable(avl_tree_app main.cpp)
   - target_link_libraries(avl_tree_app PRIVATE avl_tree_lib)
```

**Verify:** confirm all three files exist and the library target uses `PUBLIC` for the include directory.

**Commit:**

```bash
git add src/CMakeLists.txt tests/unit/CMakeLists.txt app/CMakeLists.txt
git commit -m "Step 6.3: subdirectory CMakeLists.txt files"
```

### Sub-prompt 6.4: Configure the build

*Recommended mode: Act mode. The action is a single known command.*

```text
Run the cmake configure command from the project root:

  cmake -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

The configure step must complete without error. It will download GoogleTest
on first run, which takes a few minutes. Report the final configure summary
as confirmation.

Do not run a build yet.
```

**Verify:**

```bash
ls build/CMakeCache.txt
ls build/compile_commands.json
```

Both should exist.

> **Important: do not attempt a full build (`cmake --build build/`) at this stage.** All five source files are still empty placeholders, but two of them have different consequences:
>
> - `src/avl_tree.cpp` is empty: the library target `avl_tree_lib` will build (producing an empty `.a` file), and `tests/unit/avl_tree_unit_tests` will fail to link because no AVLTree symbols exist. This is expected and is the point of sub-prompt 7.4.
> - `app/main.cpp` is empty: `avl_tree_app` will fail to link with "Undefined symbols: _main" because the linker cannot find a `main()` entry point. This is fixed in sub-prompt 8.4.
>
> Triggering either failure now wastes Cline cycles as the model loops trying to "fix" the expected error. Wait until the relevant sub-prompt before invoking a build.

If configure fails, check Troubleshooting for "CMake configure fails" before proceeding.

**Commit:**

```bash
git status
# build/ is gitignored, so likely nothing to commit. If anything else was
# modified, commit it now.
```

### Squash-merge the step

```bash
git checkout main
git merge --squash step-6-infra
git commit -m "Step 6: project infrastructure (CMake, GoogleTest, directory layout)"
git branch -D step-6-infra
```

`main` now has one clean commit for the whole step. Discarded sub-step commits remain only on the deleted branch.

---

## Step 7: Interface and Unit Tests

This step defines the public contract of the AVL tree class and the tests that verify it, before any implementation exists. The correct sequence is: declare the interface, write tests against it, and confirm the tests compile but fail at link time (unresolved symbols). A *compile* error at this stage means the header or test syntax is wrong; that must be fixed before proceeding.

### Branch for the step

```bash
git checkout -b step-7-interface
```

### Sub-prompt 7.1: Write the interface header

*Recommended mode: Act mode. Review the proposed header carefully before approving; the interface defines the contract for everything that follows.*

```text
Write include/avl_tree/avl_tree.h. Do not modify any other files.

Declare class AVLTree with this public API:
  void insert(const std::string& key, int value);
    insert a new key/value pair, or update value if key exists
  std::optional<int> search(const std::string& key) const;
    return the value for key, or std::nullopt if absent
  std::vector<std::pair<std::string, int>> in_order() const;
    return all pairs sorted ascending by key
  int height() const;
    return the height of the tree (0 for empty)

Use a private nested struct Node for tree nodes (do not define members yet,
just declare the struct as private).

Use #pragma once at the top.
Include only standard headers needed by the declarations.
Do not define any method bodies; declarations only.
```

**Verify:** open the header and confirm there are no method bodies (only `;` after each signature), `#pragma once` is at the top, and `std::optional` and `std::vector` headers are included.

**Commit:**

```bash
git add include/avl_tree/avl_tree.h
git commit -m "Step 7.1: AVLTree public interface header"
```

### Sub-prompt 7.2: Write the unit tests

*Recommended mode: Act mode. Review each proposed write before approving.*

```text
Write tests/unit/avl_tree_test.cpp. Do not modify any other files.

Include "avl_tree/avl_tree.h" and <gtest/gtest.h>.

Write exactly these six TEST cases, with a brief comment above each
explaining what it verifies:

1. EmptyTreeReturnsEmptyInOrder
   AVLTree t; assert t.in_order().empty()

2. InOrderReturnsSortedPairs
   Insert several pairs; assert in_order() returns them sorted by key.

3. SearchFindsExistingKey
   Insert pairs; assert search() returns the correct value.

4. SearchReturnsNulloptForMissingKey
   Insert pairs; assert search("not-present") == std::nullopt.

5. InsertOnExistingKeyUpdatesValue
   Insert ("a", 1); insert ("a", 99); assert search("a") returns 99.

6. BalancedAfterSortedInsert
   Insert keys "a" through "g" in alphabetical order.
   Assert height() <= 4 (ceil(log2(7)) + 1 = 4 for a balanced tree of 7 nodes).

Use TEST() macros, not TEST_F(). No fixtures.
```

**Verify:** open the test file and confirm all six TEST cases are present with the exact names above.

**Commit:**

```bash
git add tests/unit/avl_tree_test.cpp
git commit -m "Step 7.2: unit tests against AVLTree interface"
```

### Sub-prompt 7.3: Write the functional test scaffold

*Recommended mode: Act mode. Review each proposed write before approving.*

```text
Write tests/functional/test_app.py. Do not modify any other files.

The application is specified in a later step to produce exactly this stdout
(one item per line, no trailing whitespace):
  apple: 5
  apricot: 3
  banana: 2
  cherry: 1
  date: 8
  search cherry: 1
  search mango: not found
  height: <integer up to 3>

Write pytest tests that:
- Use a session-scoped fixture to build the binary before any test runs:
    cmake --build build/ --target avl_tree_app
  invoked from the project root via subprocess.run with check=True.
  The fixture must fail the session if the build fails.
- Tests must not rebuild; they only invoke the binary.

Test cases:
- ExitCodeIsZero: invoking the binary returns returncode 0.
- FirstFiveLinesAreSortedKeyValuePairs: split stdout into lines, take the
  first 5, parse each as "key: value", assert keys are in alphabetical order.
- SearchCherryLineExact: one stdout line equals exactly "search cherry: 1".
- SearchMangoLineExact: one stdout line equals exactly "search mango: not found".
- HeightLineWithinBound: one stdout line begins with "height: " and the
  integer after the prefix is <= 3.

Use subprocess.run with capture_output=True, text=True, check=False
(so failure inspection is possible).
The "search cherry" assertion must match the exact line, not just a substring
containing "cherry: 1" (which would also match the in_order line).
```

**Verify:** open the test file and confirm the fixture is `scope="session"`, that `subprocess.run` uses `text=True`, and that the search assertions check for exact line matches rather than substring containment.

**Commit:**

```bash
git add tests/functional/test_app.py
git commit -m "Step 7.3: functional test scaffold (binary not yet implemented)"
```

### Sub-prompt 7.4: Confirm the tests link-fail as expected

*Recommended mode: Act mode. The action is a single build command with a known expected failure mode.*

```text
Attempt to build the unit tests:

  cmake --build build/ --target avl_tree_unit_tests

Expected outcome: a linker error reporting unresolved symbols for AVLTree
methods (since src/avl_tree.cpp is still empty). This confirms the header
and test syntax are valid.

If you instead get a compile error, the header or test file has a syntax
problem. Report the compile error so I can correct it.

Report the linker error output as confirmation.
```

**Verify:** the build output should contain "Undefined symbols" or "undefined reference to" messages naming `AVLTree::insert`, `AVLTree::search`, etc. If it says "fatal error" or names a syntax issue, stop and fix the header before proceeding.

**Commit (nothing should have changed in source files):**

```bash
git status   # should report nothing to commit
```

### Squash-merge the step

```bash
git checkout main
git merge --squash step-7-interface
git commit -m "Step 7: AVLTree interface and unit/functional tests (link-fail confirmed)"
git branch -D step-7-interface
```

---

## Step 8: Implementation and Functional Tests

This step implements the tree, gets the unit tests green, writes the host application, and gets the functional tests green.

### Branch for the step

```bash
git checkout -b step-8-impl
```

### Sub-prompt 8.1: Define the Node struct and implement basic helpers

*Recommended mode: Act mode. Review each proposed write before approving.*

```text
Open src/avl_tree.cpp. Implement only the following so far:

1. Inside the AVLTree class definition (in the header), add the Node struct
   members:
     std::string key;
     int value;
     int height{1};
     std::unique_ptr<Node> left;
     std::unique_ptr<Node> right;
     Node(std::string k, int v);

   Add #include <memory> and #include <utility> to the header if not present.

2. Add a std::unique_ptr<Node> root_ private member to AVLTree.

3. In src/avl_tree.cpp, implement:
   - The Node constructor (move key into place, store value)
   - A static int _height(const Node* n) that returns n ? n->height : 0
   - A static void _fix(Node* n) that sets n->height = 1 + max(_height(left),
     _height(right)). Add #include <algorithm>.

Do not implement any other methods yet.
```

**Verify:**

```bash
cmake --build build/ --target avl_tree_lib
# Should compile cleanly (only the helpers exist so far; no link step here).
```

**Commit:**

```bash
git add include/avl_tree/avl_tree.h src/avl_tree.cpp
git commit -m "Step 8.1: Node struct and height helpers"
```

### Sub-prompt 8.2: Implement insert with rebalancing

*Recommended mode: Act mode. Review the proposed rotation logic carefully before approving; this is the hardest part of the project and the most likely to be subtly wrong.*

```text
In src/avl_tree.cpp, implement AVL rebalancing and the insert path.
Do not modify the header except to declare any new static helpers needed.

Implement these static helpers in the .cpp file:
- int _bf(const Node* n)
    returns _height(n->left.get()) - _height(n->right.get())
- std::unique_ptr<Node> _rotate_right(std::unique_ptr<Node> z)
- std::unique_ptr<Node> _rotate_left(std::unique_ptr<Node> z)
- std::unique_ptr<Node> _rebalance(std::unique_ptr<Node> n)
    handles all four cases:
      Left-Left:    bf > 1 and _bf(left) >= 0   -> rotate right
      Left-Right:   bf > 1 and _bf(left) < 0    -> rotate left on left, then right
      Right-Right:  bf < -1 and _bf(right) <= 0 -> rotate left
      Right-Left:   bf < -1 and _bf(right) > 0  -> rotate right on right, then left
- std::unique_ptr<Node> _insert(std::unique_ptr<Node> n,
                                const std::string& key, int value)

Then implement the public methods:
- void AVLTree::insert(...) which assigns root_ = _insert(std::move(root_), ...)
- int AVLTree::height() const which returns _height(root_.get())

Declare any new helpers in the header as private static methods.
```

**Verify:**

```bash
cmake --build build/ --target avl_tree_lib
# Should compile cleanly.
```

**Commit:**

```bash
git add include/avl_tree/avl_tree.h src/avl_tree.cpp
git commit -m "Step 8.2: insert with AVL rebalancing"
```

### Sub-prompt 8.3: Implement search and in_order

*Recommended mode: Act mode. Review each proposed write before approving.*

```text
In src/avl_tree.cpp, implement the remaining public methods.
Declare any new private static helpers in the header.

Implement:
- static std::optional<int> _search(const Node* n, const std::string& key)
    recursive; returns std::nullopt if n is null or key not found
- static void _in_order(const Node* n,
                        std::vector<std::pair<std::string, int>>& out)
    appends in_order pairs to out

Public methods:
- std::optional<int> AVLTree::search(const std::string& key) const
- std::vector<std::pair<std::string, int>> AVLTree::in_order() const
```

**Verify:**

```bash
cmake --build build/ --target avl_tree_unit_tests
ctest --test-dir build/ --output-on-failure
# All six unit tests should pass. If any fail, fix the implementation;
# do not modify the tests.
```

**Commit:**

```bash
git add include/avl_tree/avl_tree.h src/avl_tree.cpp
git commit -m "Step 8.3: search and in_order; all unit tests pass"
```

### Sub-prompt 8.4: Write the host application

*Recommended mode: Act mode. Review the proposed output formatting carefully before approving; functional tests check exact line content.*

```text
Write app/main.cpp. Do not modify any other files.

The application must produce exactly this stdout (one line per item,
no extra whitespace, no trailing blank lines):
  apple: 5
  apricot: 3
  banana: 2
  cherry: 1
  date: 8
  search cherry: 1
  search mango: not found
  height: <integer>

Where <integer> is the height returned by tree.height().

Implementation:
- Include "avl_tree/avl_tree.h", <iostream>, <cstdlib>.
- Create an AVLTree.
- Insert in this order: ("banana", 2), ("apple", 5), ("cherry", 1),
  ("date", 8), ("apricot", 3).
- Iterate tree.in_order() and print each pair as "key: value\n" to std::cout.
- For tree.search("cherry"): print "search cherry: " then *result << "\n".
- For tree.search("mango"): print "search mango: not found\n".
- Print "height: " << tree.height() << "\n".
- Call std::cout << std::flush before returning EXIT_SUCCESS.
- Use std::cout, not printf. Use "\n" consistently; do not mix endl and "\n".
```

**Verify:**

```bash
cmake --build build/ --target avl_tree_app
./build/app/avl_tree_app
# Output should match the spec exactly, eight lines.
```

**Commit:**

```bash
git add app/main.cpp
git commit -m "Step 8.4: host application"
```

### Sub-prompt 8.5: Run the functional tests

*Recommended mode: Act mode. The action is a single test command.*

```text
Run the functional test suite:

  .venv/bin/pytest tests/functional/test_app.py -v

All five tests must pass. Report the full pytest output.
If any test fails, do not modify the test. Examine the actual vs expected
output, and propose a fix to either main.cpp or the implementation.
```

**Verify:** the pytest output should show five passing tests with no errors.

**Commit (no source changes expected if all passed):**

```bash
git status
# Likely nothing to commit. If any fix was needed and made, commit it:
git add -A
git commit -m "Step 8.5: functional tests pass"  # only if changes were made
```

### Squash-merge the step

```bash
git checkout main
git merge --squash step-8-impl
git commit -m "Step 8: AVLTree implementation, host app, all tests pass"
git branch -D step-8-impl
```

You now have a working AVL tree with both test layers green, and three clean commits on `main` representing the journey from empty project to working implementation.

---

## Step 9: Reference Implementation

Use this to cross-check the structure and rotation logic of Cline's output. The unit tests are the definitive measure of correctness, not similarity to this listing.

**include/avl_tree/avl_tree.h**

```cpp
#pragma once
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

class AVLTree {
public:
    void insert(const std::string& key, int value);
    bool remove(const std::string& key);
    std::optional<int> search(const std::string& key) const;
    std::vector<std::pair<std::string, int>> in_order() const;
    int height() const;

private:
    struct Node {
        std::string key;
        int value;
        int height{1};
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        Node(std::string k, int v) : key(std::move(k)), value(v) {}
    };

    std::unique_ptr<Node> root_;

    static int _height(const Node* n);
    static int _bf(const Node* n);
    static void _fix(Node* n);
    static std::unique_ptr<Node> _rotate_right(std::unique_ptr<Node> z);
    static std::unique_ptr<Node> _rotate_left(std::unique_ptr<Node> z);
    static std::unique_ptr<Node> _rebalance(std::unique_ptr<Node> n);
    static std::unique_ptr<Node> _insert(std::unique_ptr<Node> n,
                                          const std::string& key, int value);
    static std::unique_ptr<Node> _remove(std::unique_ptr<Node> n,
                                          const std::string& key, bool& removed);
    static std::unique_ptr<Node> _detach_min(std::unique_ptr<Node> n,
                                              std::unique_ptr<Node>& out_min);
    static std::optional<int> _search(const Node* n, const std::string& key);
    static void _in_order(const Node* n,
                           std::vector<std::pair<std::string, int>>& out);
};
```

**src/avl_tree.cpp**

```cpp
#include "avl_tree/avl_tree.h"
#include <algorithm>

int AVLTree::_height(const Node* n) { return n ? n->height : 0; }

int AVLTree::_bf(const Node* n) {
    return _height(n->left.get()) - _height(n->right.get());
}

void AVLTree::_fix(Node* n) {
    n->height = 1 + std::max(_height(n->left.get()), _height(n->right.get()));
}

std::unique_ptr<AVLTree::Node> AVLTree::_rotate_right(std::unique_ptr<Node> z) {
    auto y = std::move(z->left);
    z->left = std::move(y->right);
    _fix(z.get());
    y->right = std::move(z);
    _fix(y.get());
    return y;
}

std::unique_ptr<AVLTree::Node> AVLTree::_rotate_left(std::unique_ptr<Node> z) {
    auto y = std::move(z->right);
    z->right = std::move(y->left);
    _fix(z.get());
    y->left = std::move(z);
    _fix(y.get());
    return y;
}

std::unique_ptr<AVLTree::Node> AVLTree::_rebalance(std::unique_ptr<Node> n) {
    _fix(n.get());
    int bf = _bf(n.get());
    if (bf > 1) {
        if (_bf(n->left.get()) < 0)
            n->left = _rotate_left(std::move(n->left));
        return _rotate_right(std::move(n));
    }
    if (bf < -1) {
        if (_bf(n->right.get()) > 0)
            n->right = _rotate_right(std::move(n->right));
        return _rotate_left(std::move(n));
    }
    return n;
}

std::unique_ptr<AVLTree::Node> AVLTree::_insert(
        std::unique_ptr<Node> n, const std::string& key, int value) {
    if (!n) return std::make_unique<Node>(key, value);
    if (key < n->key)
        n->left = _insert(std::move(n->left), key, value);
    else if (key > n->key)
        n->right = _insert(std::move(n->right), key, value);
    else
        n->value = value;
    return _rebalance(std::move(n));
}

// Detach and return the minimum node from the subtree rooted at n.
// The remaining subtree (with min removed and rebalanced) is the return value.
// The detached min node is moved into out_min.
std::unique_ptr<AVLTree::Node> AVLTree::_detach_min(
        std::unique_ptr<Node> n, std::unique_ptr<Node>& out_min) {
    if (!n->left) {
        out_min = std::move(n);
        auto right = std::move(out_min->right);
        return right;   // may be null
    }
    n->left = _detach_min(std::move(n->left), out_min);
    return _rebalance(std::move(n));
}

std::unique_ptr<AVLTree::Node> AVLTree::_remove(
        std::unique_ptr<Node> n, const std::string& key, bool& removed) {
    if (!n) { removed = false; return nullptr; }
    if (key < n->key) {
        n->left = _remove(std::move(n->left), key, removed);
    } else if (key > n->key) {
        n->right = _remove(std::move(n->right), key, removed);
    } else {
        removed = true;
        if (!n->left)  return std::move(n->right);
        if (!n->right) return std::move(n->left);
        // Two children: replace n with its in-order successor.
        std::unique_ptr<Node> successor;
        auto new_right = _detach_min(std::move(n->right), successor);
        successor->left = std::move(n->left);
        successor->right = std::move(new_right);
        n = std::move(successor);
    }
    return _rebalance(std::move(n));
}

std::optional<int> AVLTree::_search(const Node* n, const std::string& key) {
    if (!n) return std::nullopt;
    if (key == n->key) return n->value;
    const auto& next = key < n->key ? n->left : n->right;
    return _search(next.get(), key);
}

void AVLTree::_in_order(const Node* n,
                         std::vector<std::pair<std::string, int>>& out) {
    if (!n) return;
    _in_order(n->left.get(), out);
    out.emplace_back(n->key, n->value);
    _in_order(n->right.get(), out);
}

void AVLTree::insert(const std::string& key, int value) {
    root_ = _insert(std::move(root_), key, value);
}

bool AVLTree::remove(const std::string& key) {
    bool removed = false;
    root_ = _remove(std::move(root_), key, removed);
    return removed;
}

std::optional<int> AVLTree::search(const std::string& key) const {
    return _search(root_.get(), key);
}

std::vector<std::pair<std::string, int>> AVLTree::in_order() const {
    std::vector<std::pair<std::string, int>> out;
    _in_order(root_.get(), out);
    return out;
}

int AVLTree::height() const { return _height(root_.get()); }
```

Note: the `remove` method shown here is the target for the iteration exercise in Step 10. The header and source above include it for reference; the version Cline produces in Step 8 will not have it yet.

---

## Step 10: Iterating with Cline

### Adding features

Use a **new task** for each distinct feature. Starting fresh discards accumulated history and gives the model a clean working memory. Supply context explicitly; do not rely on Cline to recall it.

**Example: add a `delete` method** (new task):

```text
The project is an AVL tree in C++20.
Interface: include/avl_tree/avl_tree.h
Implementation: src/avl_tree.cpp
Unit tests: tests/unit/avl_tree_test.cpp

Add a method:
  bool remove(const std::string& key);
  Removes the node with the given key, rebalances, and returns true.
  Returns false if the key is not present.

Add this declaration to the header and implement it in the .cpp file.
After implementing, add unit tests for remove() to avl_tree_test.cpp covering:
  - remove returns true for a present key
  - remove returns false for an absent key
  - in_order() no longer contains the removed key after removal
  - the tree remains balanced after removal (height() <= expected ceiling)

Build and run all unit tests:
  cmake --build build/ && ctest --test-dir build/ --output-on-failure

All tests must pass before finishing.
```

### Correcting errors

Anchor corrections to specific failing test output:

```text
The unit test BalancedAfterSortedInsert is failing:
  Expected height() <= 4, got 6

The _rebalance function in src/avl_tree.cpp is not triggering the Left-Right
rotation case correctly. Fix only _rebalance; do not change any other function.
```

### Checkpoints and rollback

Cline snapshots the working directory before each Act step. If a step corrupts a file, find the last known-good checkpoint in the Cline panel and roll back without manual `git checkout`.

---

## Troubleshooting

### A note on staying unstuck

This tutorial was written at the start of June 2026. The local LLM ecosystem moves quickly: Ollama, Cline, MLX, and the underlying models all release updates on a timescale of weeks rather than months. By the time you read this, command-line flags may have changed, error messages may have been reworded, default settings may differ, and entire components may have been deprecated or renamed. Individual configurations vary too: macOS version, Homebrew version, what other tools are installed, what shell is in use, prior environment state.

The troubleshooting entries below cover problems encountered during the tutorial's own end-to-end testing. If you hit something not listed, or a listed fix does not work, frontier LLMs are excellent debugging partners for this kind of problem. They have broad knowledge of the toolchain, can read error messages accurately, and can suggest diagnostic steps you might not think of.

Three options worth knowing, at the time of writing:

- **[Claude](https://claude.ai)** (Anthropic). A free tier with daily limits is available; Pro and Max subscriptions provide higher limits, Projects, and longer context. Strong at multi-step technical reasoning and reading long error logs.
- **[Gemini](https://gemini.google.com)** (Google). Generally free for everyday use; capable on technical questions. The API via [AI Studio](https://aistudio.google.com) also has a usable free tier.
- **[Microsoft Copilot](https://copilot.microsoft.com)**. In our experience during tutorial testing, Copilot tended to hallucinate command flags, invent file paths, and confidently lead down dead ends on macOS-specific issues. Your mileage may vary, and tuning the prompt (asking for citations, requesting that it admit uncertainty, narrowing the scope) sometimes helps. For technical troubleshooting we have generally found the other two more reliable.

To get useful answers from any frontier model, paste verbatim rather than paraphrasing: the exact command you ran, the full error output (not just the headline message), and what step of the tutorial you are on. The more concrete, the better. Frontier models are also good at reading the tutorial itself: pasting both the relevant section of this document and your error often produces better answers, because the model can reason about both the intent and the failure mode together.

**Diagnostic snippet.** Run this block in a terminal and paste the entire output into your chat session along with your question. It captures every version and state detail relevant to this stack:

```bash
echo "=== macOS ==="; sw_vers
echo "=== Homebrew ==="; brew --version
echo "=== Ollama ==="; ollama --version; which ollama
echo "=== Loaded models ==="; ollama ps
echo "=== Installed models ==="; ollama list
echo "=== Cline extension ==="; code --list-extensions --show-versions 2>/dev/null | grep claude-dev || echo "(code CLI not available; check version in VS Code Extensions panel)"
echo "=== Python ==="; python3 --version; which python3
echo "=== CMake ==="; cmake --version | head -1
echo "=== Shell ==="; echo "$SHELL  ($ZSH_VERSION$BASH_VERSION)"
echo "=== Ollama service status ==="; brew services info ollama 2>/dev/null | head -5
```

---

### Common issues

**Cline reports "Invalid URL" error connecting to Ollama**
Check Cline's settings for the **Ollama API Key** field. It must be empty. Local Ollama has no authentication; anything in this field (even non-printing characters) causes Cline to construct a malformed URL with embedded credentials. The help text in the settings panel confirms: "Leave empty for local installations." Also verify the Base URL field contains no trailing whitespace; retype it by hand if pasted.

**Cline reports "Ollama request timed out after 30 seconds"**
This is almost always a cold-start issue, not a connection problem. The first request after Ollama loads a model from cold can take 30 to 90 seconds on a 24GB machine, longer than Cline's default 30-second timeout. Raise the **Request Timeout (ms)** setting in Cline to `300000` (5 minutes). Subsequent requests are fast because the model stays resident; the timeout only matters for cold starts.

While the request is pending, you can confirm Ollama is making progress with `ollama ps` in a separate terminal — once the model appears in the output, it has finished loading.

**Cline cannot connect to Ollama**
Confirm Ollama is running (`ollama list` should respond) and the base URL is `http://localhost:11434`. If you used `brew services start ollama`, check with `brew services list`.

**Ollama starts but `ollama list` returns nothing or fails**
If you previously ran `sudo ollama serve` or otherwise started Ollama as a different user, the model directory `~/.ollama` may have wrong ownership. Check with:

```bash
ls -la ~/.ollama
```

If owned by root or another user, restore ownership:

```bash
sudo chown -R $(whoami):staff ~/.ollama
```

**Cline and Ollama context window settings disagree**
The smaller of the two values wins. Ollama will not honour a request for more tokens than `num_ctx` in the Modelfile, and Cline will not send more than its own Context Window setting. If you set the Modelfile to 16384 but leave Cline at the default 8192, you get 8192 effective context. Symptoms: Cline truncates file content unexpectedly, or refuses to read a file it had no trouble reading in earlier sessions. Confirm both settings match.

**Model not found / unknown model in Cline**
The `gemma3-cline` variant must be created before it can be used. If missing:

```bash
ollama create gemma3-cline -f ~/ollama-models/gemma3-cline.modelfile
```

**Ollama version below 0.19 (no MLX acceleration)**

```bash
brew upgrade ollama
# Then recreate your model variant:
ollama create gemma3-cline -f ~/ollama-models/gemma3-cline.modelfile
```

**CMake configure fails after Step 6**
Common causes:

- CMake version below 3.24. Check with `cmake --version`.
- No internet access for FetchContent to download GoogleTest from GitHub. FetchContent requires network access on first configure.
- Corporate proxy or HTTPS interception breaking the GitHub fetch. Symptoms: TLS handshake errors, "could not resolve host", or "SSL certificate problem" in the cmake output. Fixes vary by environment; typically setting `HTTPS_PROXY` and `HTTP_PROXY` environment variables, or pre-installing your organisation's CA certificate. If proxy access is not available, install GoogleTest via Homebrew (`brew install googletest`) and modify the root `CMakeLists.txt` to use `find_package(GTest REQUIRED)` instead of FetchContent.

**Unit tests compile but fail unexpectedly**
Check that `avl_tree.cpp` is not empty (a common state after the Step 6 infrastructure prompt). If it contains only a placeholder comment, the linker will report unresolved symbols for every method. This is expected until Step 8's implementation is complete.

**Functional tests fail with "binary not found"**
The pytest fixture builds `avl_tree_app` via cmake before tests run. If it reports the binary is missing, confirm the cmake build succeeded and the binary exists at `build/app/avl_tree_app`. If the path differs, update the path in `test_app.py`.

**Generation is slow or the machine runs hot**
Open Activity Monitor and check Memory Pressure. Yellow or red, or rising Swap Used, means context has grown too large. Start a new Cline task. If it persists, reduce `num_ctx` to 8192:

```bash
# Edit ~/ollama-models/gemma3-cline.modelfile and change num_ctx to 8192
ollama create gemma3-cline -f ~/ollama-models/gemma3-cline.modelfile
```

**Model cold-reload lag between Cline steps**
By default Ollama unloads a model after 5 minutes of inactivity. To keep it resident, stop the brew service and run Ollama in foreground from a dedicated terminal tab: `brew services stop ollama` then `OLLAMA_KEEP_ALIVE=-1 ollama serve`. Setting `OLLAMA_KEEP_ALIVE` in `~/.zshrc` does **not** work for the brew background service because launchd does not read shell startup files. See "Keep the Model Loaded Between Steps" in Step 2. Verify with `ollama ps`: the UNTIL column should read "Forever" rather than a countdown.

**Build fails with "Undefined symbols: _main" or "undefined reference to AVLTree::..."**
These failures are expected at certain stages of the tutorial and are not bugs:

- `Undefined symbols: _main` (or equivalent) for `avl_tree_app`: the `app/main.cpp` file is still an empty placeholder. The linker cannot produce an executable without a `main()` function. This is resolved in sub-prompt 8.4.
- Undefined references to `AVLTree::insert`, `AVLTree::search`, etc. for `avl_tree_unit_tests`: the `src/avl_tree.cpp` file is still empty. The implementation arrives in sub-prompts 8.1-8.3. The link failure of the unit-test target between Steps 7 and 8 is intentional and is the point of sub-prompt 7.4.

If Cline encounters either error and begins iterating to fix it, cancel the task. The model cannot fix what the tutorial intentionally defers. Tell Cline (in a new task) which target should be built next, e.g. `cmake --build build/ --target avl_tree_unit_tests`, rather than running the unqualified `cmake --build build/` which builds everything.

**`brew services info ollama` shows "Running: ✘" but Ollama responds normally**
This is expected if you are using the foreground `ollama serve` approach. Your foreground process owns port 11434; the brew-managed launchd service is genuinely stopped. Both are correct. The check that actually matters is whether Ollama responds to requests: `curl http://localhost:11434/api/tags` should return JSON listing your models. Do not run `brew services start ollama` while foreground Ollama is running; it will fail with a port conflict.

**Model produces broken code or repeats itself**
Keep tasks small. Approve each step individually. If the model loops, cancel immediately. Use failing test output as the correction prompt rather than "try again".

**Cline created files where directories should be (or write_to_file fails with ENOTDIR)**
Symptom: errors like `ENOTDIR: not a directory, lstat '...path/to/file'`, or `mkdir` later reports "File exists" for what should be a directory. This happens when Cline's `write_to_file` is invoked on a directory path. The tool creates a regular file at that path; subsequent attempts to create files inside it fail because you cannot put a file inside a file. Recovery on a 9-12B model takes 5 to 10 minutes of re-prompting, so prevention is preferred.

Diagnose what got created where it should not be:

```bash
find . -maxdepth 3 -type f -not -path "./.git/*" -not -path "./.venv/*"
file include/avl_tree src tests/unit tests/functional app 2>/dev/null
```

Anything reported as "empty" or "ASCII text" where you expected "directory" is the problem. Recovery, in increasing order of severity:

```bash
# Targeted: remove the file-shaped pseudo-directory and create the real directory
rm path/to/wrong/file
mkdir -p path/to/wrong/file

# Step-level: abandon the current branch and start the step over
git checkout main
git branch -D step-N-<tag>
git checkout -b step-N-<tag>
# Re-run the sub-prompts from the start of the step
```

Prevention: the updated `.clinerules` template in Step 5 instructs the model to use `mkdir -p` via `execute_command` for directories, not `write_to_file`. If you adapted `.clinerules` and removed that rule, add it back. Sub-prompt 6.1 also splits the task into explicit Phase 1 (mkdir) / Phase 2 (write files) to reinforce the distinction.

**`python` or `pytest` ignores the venv**
Ensure the venv is active in VS Code's integrated terminal (`source .venv/bin/activate`). Use `.venv/bin/pytest` explicitly in all Cline-facing commands.

**`ollama pull` fails with `Error: EOF` after an interrupted download**
When a download is interrupted (network drop, Ctrl+C, machine sleep), Ollama leaves the partial blob plus its parallel-chunk metadata in the cache. Subsequent pulls fail at the manifest stage with `Error: EOF` because the daemon tries to resume from stale chunk offsets that no longer match the registry's view of the file.

Ollama downloads large blobs in roughly 16 parallel chunks, so a single interrupted pull leaves the main `-partial` file (potentially several GB) plus 16 small `-partial-N` metadata files. All of them must be cleared:

```bash
# Confirm what's present:
ls -la ~/.ollama/models/blobs/ | grep partial

# Stop the service (the daemon holds locks on these files):
brew services stop ollama

# Remove all partials. The trailing * matches both the bare -partial
# file and the numbered -partial-N chunks:
rm ~/.ollama/models/blobs/*partial*

# Confirm cleared and restart:
ls -la ~/.ollama/models/blobs/ | grep partial   # should produce no output
brew services start ollama
ollama pull qwen3.5:9b
```

You lose whatever was already downloaded and start over; there is no reliable way to splice an orphaned partial back into a fresh pull. If you anticipate further interruptions, consider running the pull foreground via `ollama pull` in a screen/tmux session rather than risking a SIGINT.

If pulls still fail at the manifest stage after clearing partials, the issue is upstream rather than local. Test the registry directly:

```bash
curl -I https://registry.ollama.ai/v2/library/qwen3.5/manifests/9b
```

Anything other than HTTP 200 or 401 (timeout, TLS error, EOF) points to a network or registry problem. Wait a few minutes and try again.

**Commands with trailing `# comments` produce no output or behave oddly**
By default, zsh does not treat `#` as a comment character at the interactive prompt; only inside scripts. If you copy a command with a trailing comment such as `cmake --version   # should report 3.24+`, zsh passes the comment text as arguments to the command, which may then silently fail or behave unexpectedly. To enable shell-style comments at the interactive prompt:

```bash
setopt interactive_comments
```

Add the line to `~/.zshrc` to make it permanent. Alternatively, strip the comment before running the command, or put it on a separate line above.

---

## Reference: `.clinerules` and Optimisation

### What `.clinerules` can and cannot do

It adds text to the system prompt, influencing scope, file access patterns, coding style, and tool usage. It cannot override fundamental model capability limits; a 9B model with perfect instructions will still occasionally hallucinate a symbol name. Think of it as guardrails, not a substitute for reviewing diffs.

### Best practices

**Be specific about scope.** Vague rules have little effect. Concrete rules ("read only the file explicitly named in the task") are reliably followed.

**Name the build commands explicitly.** Smaller models will invent plausible-looking but wrong cmake invocations. Stating `cmake --build build/` and `ctest --test-dir build/` removes the ambiguity.

**Name the environment explicitly.** Telling the model it is running on constrained hardware primes it to favour minimal approaches. Most training data comes from large cloud machines.

**Keep the test tool paths explicit.** State `.venv/bin/pytest` and `.venv/bin/python3`. Without this, Cline may invoke system Python and bypass the venv.

**Keep it short.** Rules that run to several paragraphs consume context tokens on every task. A tight list of actionable constraints outperforms an essay.

**Version-control it.** The `.clinerules` file belongs in git. It travels with the repository and documents working conventions.

**Update it when changing language or build system.** The current file is specific to C++20/CMake/pytest. If you adapt it for a Python-only project, remove the cmake lines. If you move to a different build system, update the build commands. Stale `.clinerules` content pointing at tools that do not apply to the project will confuse the model.

### Optimisation techniques

**Prompt engineering for smaller models**

- State constraints upfront: "No external dependencies. The public API is exactly four methods." Front-loading prevents the model from inventing its own.
- Provide the interface before asking for the implementation. If you know the method signatures, include them. The model fills in the body rather than designing the structure.
- Use concrete examples for ambiguous requirements. "Return pairs sorted ascending by key, e.g. `[{"a", 1}, {"b", 2}]`" is more actionable than "return a sorted list".
- Ask for one thing at a time. Compound prompts cause the model to attempt multiple changes in a single diff; one error invalidates the whole step.

**Managing context**

- Start a new task for each distinct feature. Every prior exchange and file read is tokens consumed from your budget.
- Name the specific file Cline should read in Plan mode rather than letting it scan the project.
- Watch Activity Monitor. Memory Pressure yellow means caution; red means swapping. Start a new task rather than waiting out a swap-heavy session.

**MCP servers for token-efficient tooling**

Cline supports Model Context Protocol (MCP) servers, which provide the model with additional tools beyond file read/write and terminal commands. A file-search MCP server lets Cline locate a symbol across a codebase with a single targeted query rather than reading multiple files, a meaningful token saving once the project grows beyond a few modules. See the [Cline MCP documentation](https://github.com/cline/cline#mcp) for setup details.

---

## Finding Models to Experiment With

### Comparing the recommended models

> **Last reviewed: June 2026.** Throughput figures below are drawn from published benchmarks (Ollama's MLX blog and community sources) and are calibrated to specific hardware that may differ from yours. Treat them as orders of magnitude, not precise predictions. See "Measuring on your hardware" below for the protocol to verify.

The three models recommended in Step 2 cover a spectrum of size, capability, and memory cost. The table below summarises the trade-offs.

| Model | Architecture | Active params | Size | Min RAM | Reported decode (tok/s) | Source / notes |
|---|---|---|---|---|---|---|
| Qwen 3.5 9B (Q4_K_M) | Dense | 9B | ~6.6 GB | 16 GB | ~50–80 on M4 Air | Community reports on M3/M4 with MLX backend |
| Gemma 3 12B (Q4_K_M) | Dense | 12B | ~8 GB | 24 GB | ~25–45 on M4 Air | Community reports; our own runs in this tutorial fell in this range |
| Qwen 3.5 35B-A3B (NVFP4, coding) | MoE | 3B active / 35B total | ~22 GB | 32 GB+ | ~58 (Ollama 0.18 baseline) to ~112 (Ollama 0.19 MLX, M5) | [Ollama MLX blog, March 2026](https://ollama.com/blog/mlx) |

A few things worth noting from the numbers:

- The 35B-A3B is **not slower than the 12B** at decode time, despite being almost three times larger. MoE only activates 3B parameters per token, so per-token compute is closer to a small dense model. The architecture cost is in memory, not throughput.
- Decode throughput on the 35B model varies widely (58 to 112 tok/s in the Ollama benchmarks) depending on the inference backend version. Ollama 0.19's MLX support roughly doubled the figure compared to 0.18. M5-class chips get an additional boost from the GPU Neural Accelerators. M4 Pro figures should sit between the two but are not separately published.
- Smaller models on the 48GB Pro will be faster than on the Air, because the Pro has active cooling and more GPU cores. Exact figures aren't published per chip variant.

### Choosing between them

**Qwen 3.5 9B.** Fastest of the three. Best when you want quick interactive turnaround on small tasks: short prompts, single-file edits, brief clarification questions. Less reliable on long agentic chains than the 12B (the smaller parameter count shows up in multi-step reasoning). On a 48GB machine, useful as a "quick mode" alongside the larger model.

**Gemma 3 12B.** The reference choice for the 24GB tier. Better reasoning than the 9B, especially on tasks where the agent needs to plan across multiple files or hold longer chains of logic. Slower per token than the 9B, but each response is usually closer to what you want, so fewer correction cycles overall. The model used throughout this tutorial's 24GB walkthrough.

**Qwen 3.5 35B-A3B (NVFP4, coding).** The 48GB headline. Substantially stronger than the smaller models on coding-specific tasks: it was variant-tuned by Alibaba for coding workloads, and NVFP4 quantisation preserves quality better than Q4_K_M at similar memory footprint. The MoE architecture means it punches well above its weight per active parameter. Best when you want frontier-leaning capability on a local machine and have the memory to fit it. Not a guarantee of frontier behaviour — it still loses to Claude Opus or GPT-5 on complex tasks — but it closes the gap considerably.

### Measuring on your hardware

Published throughput figures are starting points, not predictions for your machine. To get numbers you can trust, run a short standardised benchmark on each model:

```bash
# Warm-load the model (don't time this):
ollama run gemma3-cline "Hello" > /dev/null

# Now time a fixed-length prompt with a known response:
time ollama run gemma3-cline "Write a Python function that returns the first n Fibonacci numbers. Return only the function code, no explanation, no examples."
```

Repeat for each model variant. The total time divided by the output token count (visible in Ollama's logs, or estimable as "characters / 4") gives a rough decode rate. For a more rigorous comparison, time the AVL tree sub-prompts from Step 6 onwards against each model and count how many succeed without correction; this captures both speed and quality in a single metric relevant to actual use.

### Where MLX-ready models live

The **mlx-community** organisation on Hugging Face (`huggingface.co/mlx-community`) hosts approximately 4,800 pre-converted, pre-quantised models covering most major architectures (Qwen, Gemma, Llama, Mistral, DeepSeek, and others), typically within days of a new release. 4-bit variants offer the best size/performance tradeoff for a 24GB machine.

Ollama 0.19+ handles the MLX format transparently. Pull any supported model via `ollama pull` and acceleration is applied automatically.

### Pulling and configuring a new model

```bash
ollama pull llama3.2:3b

cat > ~/ollama-models/llama3-cline.modelfile << 'EOF'
FROM llama3.2:3b
PARAMETER num_ctx 16384
PARAMETER temperature 0.2
PARAMETER repeat_penalty 1.1
EOF

ollama create llama3-cline -f ~/ollama-models/llama3-cline.modelfile
```

Update the Model field in Cline's settings to `llama3-cline`, then run the existing tests to evaluate how it performs on the same tasks.

### Evaluating a new model

- **Aider Polyglot leaderboard** (`aider.chat/docs/leaderboards`): code editing across six languages; the most relevant benchmark for agentic editor use.
- **SWE-bench Verified** (`swebench.com/verified.html`): real-world GitHub issue resolution; the best proxy for multi-step agentic performance.

Neither tells you how a model will behave at 9B–12B parameters on your specific codebase. The most reliable test is to run the Step 7 and Step 8 prompts from this tutorial on the new model and count how many unit and functional tests pass without correction. See "Measuring on your hardware" above for a lightweight protocol.

### Staying current

- **r/LocalLLaMA** (`reddit.com/r/LocalLLaMA`): new releases discussed within hours, often with Apple Silicon benchmarks.
- **Simon Willison's blog** (`simonwillison.net`): hands-on notes on new models and tooling.

---

## Summary

| Component | Choice | Why |
|---|---|---|
| Agent | Cline (VS Code, publisher: saoudrizwan) | Human-in-the-loop; Checkpoints; MCP support |
| Model server | Ollama 0.19+ | Automatic MLX backend on Apple Silicon |
| Model | `gemma3-cline` (24GB) or `qwen35b-cline` (48GB) — Modelfile variants | 16K or 32K context; tuned temperature |
| Language | C++20 | Target language for the data structure |
| Build system | CMake 3.24+ | Cross-platform; CTest integration |
| Unit tests | GoogleTest via FetchContent | C++-level correctness; runs via CTest |
| Functional tests | pytest + subprocess | Binary-level correctness; independent of C++ build |
| Project rules | `.clinerules` | Scope, build commands, venv path |

---

## Reference Links

### MLX
- [MLX documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [Apple ML Research: MLX](https://machinelearning.apple.com/research/mlx)
- [mlx-community on Hugging Face](https://huggingface.co/mlx-community)
- [mlx-lm on GitHub](https://github.com/ml-explore/mlx-lm)

### Ollama
- [Ollama](https://ollama.com)
- [Ollama model library](https://ollama.com/library)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Ollama MLX announcement](https://ollama.com/blog/mlx)

### Cline
- [Cline on VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=saoudrizwan.claude-dev)
- [Cline GitHub](https://github.com/cline/cline)

### Build tooling
- [CMake documentation](https://cmake.org/documentation/)
- [GoogleTest](https://github.com/google/googletest)
- [pytest](https://docs.pytest.org)

### Model developers
- [Google DeepMind Gemma](https://ai.google.dev/gemma)
- [Alibaba Qwen](https://qwenlm.github.io)
- [Meta AI (Llama)](https://ai.meta.com/llama/)
- [Mistral AI](https://mistral.ai)

### Benchmarks and model evaluation
- [Aider Polyglot leaderboard](https://aider.chat/docs/leaderboards/)
- [SWE-bench Verified leaderboard](https://www.swebench.com/verified.html)
- [LMSYS Chatbot Arena](https://lmarena.ai)

### Community and further reading
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)
- [Simon Willison's blog](https://simonwillison.net)
- [Hugging Face model hub](https://huggingface.co/models)

---

## Appendix A: Background

### Apple Silicon, MLX, and unified memory

Apple's MLX is an open-source machine learning framework built for Apple Silicon's unified memory architecture. Conventional systems have separate CPU memory and GPU VRAM, with data copied across a PCIe bus between them. That copy is a major bottleneck for LLM inference. M-series chips share a single high-bandwidth memory pool across CPU and GPU cores, and MLX exploits this directly, avoiding the copies that slow down inference on conventional hardware.

From Ollama 0.19 (released March 2026), Ollama switched its inference engine on Apple Silicon from the Metal/llama.cpp backend to MLX. Measured decode throughput on M-series hardware is approximately 2× faster than the previous backend, with lower power draw. This matters most on a fanless MacBook Air where thermal headroom is limited. The acceleration is automatic on Ollama 0.19+; no configuration is required.

### Quantisation

Quantisation compresses model weights from full floating-point precision (typically FP16 or FP32) to fewer bits, commonly 4-bit or 8-bit integers. This reduces memory footprint and speeds up inference at a small quality cost. A 4-bit quantised 12B model occupies roughly 8 GB rather than approximately 24 GB at full precision, making it practical on 24GB hardware.

When choosing models, the suffix `Q4_K_M` or `4bit` indicates 4-bit quantisation; `Q8` or `8bit` indicates 8-bit. For a 24GB machine, 4-bit is the right choice for models above 7B parameters. For models 7B and smaller, 8-bit produces better output without exceeding the memory budget.

### Memory pressure on 24GB

On a 24GB machine the system uses approximately 4 to 6 GB at idle, leaving around 18 to 20 GB for everything else. A 4-bit 12B model occupies roughly 8 GB; KV cache for a 16K context adds another 1 to 2 GB. With `OLLAMA_KEEP_ALIVE=-1` keeping the model resident, you have roughly 8 to 10 GB available for VS Code, Chrome, and anything else. macOS swaps to SSD once unified memory is exhausted, and at that point generation speed drops sharply.

This constraint shapes every recommendation in this document: it is why `num_ctx` is set to 16384 rather than 64K, why 4-bit quantisation is preferred over 8-bit, and why the `.clinerules` file discourages directory-wide reads.

### Why two test layers

The project uses two complementary test layers with different purposes.

**GoogleTest (C++)** covers unit behaviour at the library level, that is, correctness of the AVL tree logic itself. These tests compile and link against the implementation directly and run via CTest. They are fast, give specific failure messages, and can test internal invariants such as balance height that are not visible from outside the library.

**pytest (Python)** covers functional behaviour at the binary level: does the compiled application produce the correct output when invoked as a process? These tests use Python's `subprocess` module to run the compiled binary and assert on stdout. They provide an end-to-end signal independent of the C++ build system, and are easy to extend with new scenarios without recompiling. They catch issues the unit tests do not: incorrect output formatting, broken stdout flushing, exit code handling, command-line argument parsing.

The venv exists solely for pytest and its dependencies. It is not used by the C++ build.

### Why AVL trees for the worked example

An AVL tree is a self-balancing binary search tree. After every insert it checks whether the tree has become unbalanced (one subtree taller than the other by more than 1) and performs rotations to restore balance, guaranteeing O(log n) search and insert times regardless of insertion order. Without rebalancing, a plain BST built from sorted input degrades into a linked list with O(n) operations.

The four rebalance cases (Left-Left, Right-Right, Left-Right, Right-Left) are named for the path from the unbalanced node to the new node. Each requires one or two rotations to restore the AVL invariant: every node's two subtrees differ in height by at most 1.

This makes a good worked example: the algorithm is well-defined, the correctness criteria are precise (tests can verify balance via tree height), and the implementation fits within a constrained context window. Other balanced BST variants (Red-Black trees, B-trees, skip lists) are valid alternatives but involve more complex invariants. AVL is the clearest starting point.

### Model choice

Both recommended models (`gemma3:12b` and `qwen3.5:9b`) are supported by the Ollama MLX backend. Qwen 3.5 9B offers a larger native context window (256K vs 128K tokens), which suits longer agentic sessions. Gemma 3 12B's larger parameter count generally yields stronger reasoning on complex tasks. The tutorial uses `gemma3:12b` throughout; switching is straightforward (create an equivalent Modelfile and update Cline's Model setting).

---

*Draft, May 2026.*
