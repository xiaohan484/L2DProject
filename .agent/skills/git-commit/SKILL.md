---
name: git_commit_generator
description: 當使用者要求撰寫 git commit message 時，必須呼叫此技能。
---

# Commit Message 生成規則

當被要求產生 Commit Message 時，請遵循以下邏輯：

1.  **分析變更**：讀取 `git diff` 或使用者提供的程式碼變更。
2.  **格式規範**：
    <type>(<scope>): <subject>
    (空行)
    <body>

3.  **Type 列表**：
    - feat: 新功能
    - fix: 修補錯誤
    - docs: 文件修改
    - refactor: 重構 (無功能變更)

4.  **輸出範例**：
    fix(auth): correct token validation logic

    Update the JWT verification to handle expired tokens correctly.