---
name: skill-creator
description: Use when creating, reviewing, or improving an OpenCode Agent Skill, including SKILL.md structure, trigger descriptions, workflows, resources, and validation.
license: Apache-2.0
metadata:
  source: https://github.com/anthropics/skills
---

# Skill Creator

Create focused, discoverable skills that encode repeatable procedural knowledge.

## Workflow

1. Establish the task the skill enables, its trigger situations, expected output, and important exclusions.
2. Search existing project instructions and workflows before adding duplicated guidance.
3. Choose a lowercase hyphenated name no longer than 64 characters.
4. Create `<skill-name>/SKILL.md` with required `name` and `description` frontmatter.
5. Put both what the skill does and when it should trigger in the description. Include likely literal user terms and file names.
6. Keep the body imperative, focused, and self-contained. Explain why constraints matter.
7. Move large reference material or deterministic utilities into clearly referenced `references/` or `scripts/` files.
8. Test realistic matching prompts and near-miss prompts to check triggering accuracy.
9. Review the skill for unsafe commands, hidden side effects, missing dependencies, and conflicting project instructions.

Project skills belong in `.opencode/skills/<name>/SKILL.md`. Generic personal
skills belong in `~/.config/opencode/skills/<name>/SKILL.md`. Project rules that
must always apply belong in an instruction file, not a conditionally loaded skill.
