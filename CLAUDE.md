# CLAUDE.md

## Workflow

After every commit, assess what choices were made. Use small code snippets to illustrate each choice:

```python
# Original:
pad = torch.tensor(pad, ...)  # mypy loses track

# Changed to:
pad_t = torch.tensor(pad, ...)  # separate variable
```

Not full diffs - just the essential before/after for each decision.

When a pattern is repeatedly gotten wrong, add only the correct version here with extreme brevity.
