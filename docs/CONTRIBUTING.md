# ➕ Adding a New Model

[← Back to README](../README.md)

---


1. Drop the implementation in the folder matching its publication year, e.g. `models/y2026/MyModel.py`.
2. Register it and expose the framework's standard interface:

```python
from models.registry import register_model

@register_model('MyModel', expects_4d=False)
def my_model(num_classes, bands, patch_size, **kwargs):
    return MyModel(num_classes=num_classes, bands=bands, patch_size=patch_size)
```

3. Add a bibliographic entry to `MODEL_CATALOG` in `models/registry.py` so the model appears correctly in generated tables:

```python
"MyModel": {
    'full_name':   "My Model, Spelled Out",
    'paper_title': "Exact Title of the Paper",
    'paper':       "https://doi.org/...",
    'code':        "https://github.com/original-authors/MyModel",
    'year':        2026,
    'venue':       "IEEE TGRS",
},
```

4. Document any deviation from the authors' released code in a docstring at the top of the file — ports from TensorFlow, replaced CUDA kernels, changed hyperparameter defaults. This keeps every re-implementation traceable to the original work.
5. Verify it loads and profiles cleanly:

```bash
python main.py --list-models
python model_info.py MyModel
```

Models are auto-discovered from `models/`, so no import wiring is needed beyond the decorator.

---
