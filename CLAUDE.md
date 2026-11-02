# spectral-radius

- Run tests with `uv run pytest`.
  Tests marked `data` need the raw GSS cache in `data/raw/gss/` and fail
  (rather than skip) without it; deselect them with `-m 'not data'`.
- Check with `uvx ruff format`, `uvx ruff check src tests`,
  and `uvx ty check --python .venv`.
- Be explicit rather than convenient: no implicit choices in cleaning
  (keep every row and every missing code, never drop or recode silently),
  and raise rather than fall back when something isn't declared.
  Treating missing answers as null happens only downstream, in `wide`.
- Survey data lives in `src/spectral_radius/surveys/`.
  A new survey subclasses `Source` (`source.py`),
  implements `download` and `clean`,
  declares its `demographics` and `start_year`,
  maps its missing codes onto the shared `MISSING` kinds,
  assigns each variable a scale from `scales.py`,
  and gets a question set in `QUESTION_SETS`.
  Planned sources (ANES, Michigan) are noted in `surveys/__init__.py`.
- Raw downloads are cached in `data/raw/<survey>/` by `Source.fetch`;
  `data/` is gitignored,
  so copy `data/raw` into a new worktree rather than redownloading.
- Sandboxed git commands can leave an empty, read-only `.git/config.lock`.
  If git reports it could not lock its config file
  and no git process is running, remove the lock.
