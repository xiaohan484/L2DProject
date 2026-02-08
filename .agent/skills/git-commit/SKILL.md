---
name: git_smart_committer
description: Analyze changes, stage files intelligently, and commit with a standard message.
---

# 🤖 WORKFLOW: SMART COMMIT
You are an intelligent git assistant. When the user asks to "commit", follow this EXACT sequence:

## STEP 1: CHECK STATUS
1.  Run `git status`.
2.  **Decision Point**:
    * **Scenario A (Files already staged)**: Proceed to STEP 3 directly.
    * **Scenario B (No files staged, but modified files exist)**: You must STAGE files first (Go to STEP 2).
    * **Scenario C (Clean working tree)**: STOP and tell the user "Nothing to commit".

## STEP 2: STAGE FILES (The "Organize" Phase)
1.  Identify modified/untracked files from Step 1.
2.  **Action**: 
    * If the user said "Commit everything" or intent is general: Run `git add .`
    * If the user specified a scope (e.g., "Commit the login fix"): Run `git add <specific_files>`
3.  **Verification**: Run `git status` again to confirm files are staged.

## STEP 3: ANALYZE & WRITE MESSAGE
1.  Run `git diff --staged` to understand WHAT changed.
2.  Generate a commit message following the **Standard Format**:
    `<type>(<scope>): <subject>`
    
    `<body>`

## STEP 4: EXECUTE
1.  Output the final command plan to the user.
2.  Execute: `git commit -m "your_message"`

## NOTE
When running read-only commands, combine them into a single block to minimize permission prompts.

---

# ⚠️ SAFETY RULES (CRITICAL)
- **Do NOT loop**: If `git commit` fails, do NOT try to rewrite the message. Stop and report the error (usually due to file locking or hooks).
- **Context**: If you run `git add .`, explicitly tell the user: "I am staging ALL changes."