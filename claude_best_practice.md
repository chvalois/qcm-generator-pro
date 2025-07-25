#1. Use Plan Mode (Shift + Tab)
#2. Generate and use CLAUDE.md file (/init)
#3. Use Git as a checkpoint system (commit frequently)
#4. Drag screenshots to Claude (like errors)
#5. Give Claude multiple codebases (drag other folders directly to claude)
#6. Give Claude URLs for documentation
#7. Use subagents (for massive tasks, can you run subagents when necessary)
#8. Have Claude double check (can you find edge cases, etc.)
#9. Always review the code
#10. Use MCP Context7 for updated documentation

Tester les MCP Notion (pour sauvegarder les infos et les partager entre devices) et Playwright (à tester)

---

Always clean up the mess (inspect directory and clean what is not necessary but ask user always)
Cut project into different phases and each phase must enable the user to test properly what has been done through terminal or UI
KISS / YAGNI
Préciser langage du code, d'autant plus si version internationale multi langues
SRP / SOLID
Dependency Injection
Clean Architecture

Ne pas faire de fallback sans confirmation du user (souvent c'est histoire de traiter un edge case trop facilement)
---

Use Serena : 
fill mcp.json
uv run --directory /mnt/c/Users/massi/Dev/serena serena-mcp-server
in claude code : launch activate projet (use serena MCP) serena - activate_project (MCP)(project: "/mnt/c/Users/massi/Dev/qcm_master")
