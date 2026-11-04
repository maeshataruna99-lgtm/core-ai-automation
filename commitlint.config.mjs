// Commit message rules.
// Why: a consistent history makes `git log` readable and shows what each change is about.
// Used by: wagoid/commitlint-github-action in .github/workflows/commitlint.yml, which
// finds this file by its default name. The action bundles config-conventional, so the
// Python project does not need a package.json.
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // 72 keeps the subject readable in `git log --oneline` and on GitHub.
    'header-max-length': [2, 'always', 72],
    // Scopes follow the modules in docs/ARCHITECTURE.md, so a scope points at a real area.
    'scope-enum': [
      2,
      'always',
      ['api', 'services', 'llm', 'retrieval', 'agents', 'integrations', 'evaluation', 'frontend', 'docs', 'ci', 'deps'],
    ],
  },
};
