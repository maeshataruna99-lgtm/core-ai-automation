// Commit message rules, enforced in CI by .github/workflows/commitlint.yml.
export default {
  extends: ['@commitlint/config-conventional'],
  rules: {
    // 72 keeps the subject readable in `git log --oneline` and on GitHub.
    'header-max-length': [2, 'always', 72],
    'scope-enum': [
      2,
      'always',
      ['api', 'services', 'llm', 'retrieval', 'agents', 'integrations', 'evaluation', 'frontend', 'docs', 'ci', 'deps'],
    ],
  },
};
