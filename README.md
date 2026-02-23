# spectral-radius

See [here](https://arxiv.org/abs/2512.00955) for arXiv preprint.

## Replication

The following snippet will download all required code and data, run all analyses, and compile the most recent version of the working paper. You'll need `uv` and `latexmk`.

```python
git clone https://github.com/alipatti/spectral-radius &&
    cd spectral-radius &&
    uv run python -m spectral_radius &&
    latexmk paper/main -pv
```
