# Contributing to SnapSeg

Thanks for contributing to SnapSeg.
This guide defines the workflow and quality bar for all contributions.

## Development Setup

**Requirements:** Python 3.10+ (CUDA GPU recommended, CPU supported but slower)

```bash
git clone https://github.com/yorunakai/SnapSeg.git
cd SnapSeg
python -m pip install -r requirements.txt
python main.py
```

Open `http://127.0.0.1:7861` and verify the annotator loads.

Optional backend mode:

```bash
python main.py --backend mobile_sam
```

Optional checkpoint directory override:

```bash
python main.py --checkpoint-dir "C:\path\to\Model Checkpoints"
```

Model weights may be loaded from local checkpoints or Hugging Face cache depending on configuration.
Do **not** commit downloaded checkpoints.

---

## Commit Prefix Convention

Every commit message must start with one of the following prefixes:

| Prefix | When to use |
|---|---|
| `feat:` | New feature or user-visible behavior |
| `fix:` | Bug fix |
| `perf:` | Performance improvement (latency, memory, throughput) |
| `refactor:` | Code restructure with no behavior change |
| `style:` | Formatting, whitespace, naming (no logic change) |
| `docs:` | README, CONTRIBUTING, docs, or inline comment updates |
| `chore:` | Dependency updates, `.gitignore`, CI/config maintenance |
| `test:` | Adding/updating tests or profiling scripts |

Examples:

```text
feat: add brush mode for mask edge refinement
fix: correct SAM mask alignment using post_process_masks
perf: skip prefetch when free VRAM < 2 GB
docs: sync README controls section with new hotkeys
```

One concern per commit. Avoid mixed-purpose commit messages.

---

## Pull Request Checklist

Before opening a PR, verify all of the following:

### Functionality
- [ ] Feature works end-to-end (load source -> annotate -> save -> verify outputs)
- [ ] Existing flows still work: point prompt, box prompt, brush mode, autosave restore, backend fallback
- [ ] No Python exceptions in terminal during normal use
- [ ] Browser console has no new errors/warnings during key annotation flows

### Files
- [ ] No model checkpoints, no `outputs/` artifacts, no `*.pth` / `*.safetensors` / `*.bin` files staged
- [ ] No large binary files committed (use links/paths for demos where possible)
- [ ] `web/index.html` and backend API endpoints stay in sync

### Documentation
- [ ] `README.md` updated if controls, features, output format, or CLI args changed
- [ ] `README.zh-TW.md` updated to match `README.md`
- [ ] New/changed hotkeys documented in both READMEs

### Code Quality
- [ ] No hardcoded absolute paths
- [ ] No leftover debug prints in production code
- [ ] Type hints preserved on all new public functions/methods
- [ ] No new inline UI text without locale entries (`web/locales/en.json` at minimum)
- [ ] No new hardcoded UI colors outside token definitions and theme JSON files

---

## What Not to Commit

The following must never appear in a commit:

```text
# Model weights and checkpoints
*.pth
*.pt
*.safetensors
*.bin
*.ckpt

# Annotation outputs
outputs/
autosave/

# Hugging Face cache
~/.cache/huggingface/

# Runtime artifacts
__pycache__/
*.pyc
.env
```

Verify `.gitignore` before staging.
If a large file is staged by mistake, remove it with:

```bash
git rm --cached <file>
```

If a checkpoint or large artifact is already in Git history, use `git filter-repo` to purge it.
Note: this rewrites history and must be coordinated with maintainers.

---

## Code Style

SnapSeg follows standard Python conventions with project-specific rules.

### General
- Recommended line length: 100 characters
- Type hints required on all public functions and class methods
- Prefer `pathlib.Path` over `os.path` string manipulation

### Backend (`interactive_web.py`, `src/interactive/`)
- Route all session mutations through `AnnotatorSession` methods
- Hold `session.lock` for read-modify-write operations in API endpoints
- Use async managers for file writes in request paths (`AsyncSaveManager`, `AsyncAutosaveManager`)

### Frontend (`web/index.html`, `web/app.js`, `web/styles.css`)
- Keep runtime state in JS variables
- `localStorage` is allowed only for UI preferences (for example, shortcut mappings), not annotation source-of-truth
- After server-side state mutations, refresh view/state consistently (`await drawFrame()` pattern)
- New hotkeys must be wired in the keymap layer and documented in both READMEs
- New interactive handlers should be bound in `web/app.js` with `addEventListener`
- Avoid adding new inline HTML handlers (`onclick`, `onchange`); migrate existing ones when touching related UI
- New user-facing strings must be added to locale files before UI wiring
- New UI colors must use CSS tokens in `:root` and be mirrored in theme JSON files (`web/themes/*.json`)

### SAM Service (`src/interactive/sam_service.py`)
- Keep embedding cache logic centralized in `SamEmbeddingCacheService`
- Ensure mask geometry remains aligned to original image space
- Do not duplicate backend selection/caching logic outside the service layer

---

## Review Focus Areas

When reviewing PRs, pay special attention to:

- **Thread safety:** new session access outside `session.lock`
- **Mask geometry:** coordinate-space regressions after SAM pipeline changes
- **Autosave consistency:** `is_dirty` lifecycle before/after write
- **Docs sync:** README/README.zh-TW updated when UX, hotkeys, output, or CLI changes
- **Artifact hygiene:** no large files or checkpoints staged
