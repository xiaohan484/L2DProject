---
trigger: always_on
---

# 🛡️ AGENT BEHAVIOR PROTOCOL (Always Active)

## 1. MOOD DETECTION (Crucial)
You must classify the user's intent into one of two modes before responding:

* **Mode A: Architecture & Discussion** (🧠 Brainstorming)
    * **Trigger**: User asks "Why?", "How about?", "What if?", or discusses logic/trade-offs.
    * **Action**: Chat normally. **IGNORE** coding strictness constraints (like requiring unit tests in the chat). Focus on the idea.

* **Mode B: Implementation** (🛠️ Coding)
    * **Trigger**: User says "Go ahead", "Code it", "Implement", "Fix", "Refactor", or explicitly asks for code blocks.
    * **Action**: **STRICTLY ENFORCE** the Coding Standards below.

---

## 2. CODING STANDARDS (Apply ONLY in Mode B)

### A. Language Rules
* **Comments**: MUST be in **English**. (Even if the prompt is in Chinese).
* **Docstrings**: Required for all public methods.

### B. Unit Test Strategy (The Boy Scout Rule)
* **Scope**: Only generate tests for the files *currently being modified*.
* **Rule**: If you touch `logic.py`, you MUST check/update `test_logic.py`.
* **Constraint**: Do not generate tests for untouched files to save tokens.