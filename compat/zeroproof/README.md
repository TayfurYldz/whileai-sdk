# zeroproof

The ZeroProof SDK was renamed **whileai** (ZeroProof is now [While](https://while.ai)).

```bash
pip install whileai
```

```python
import whileai
import whileai.simulations as zps
```

This package is the old name. Installing it installs `whileai` and keeps
`import zeroproof` (and the older `import zeroproof_simulations`) resolving to
the very same modules, with a `DeprecationWarning`. The `zeroproof` command
still runs the CLI. `ZEROPROOF_*` environment variables and a saved
`~/.zeroproof/credentials.json` are still read by `whileai`.

Change the import when you can; this shim is not where new releases land.
Source: https://github.com/whilehq/whileai-sdk (`compat/zeroproof`).
