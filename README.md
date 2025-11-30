# spectral-radius

See [here](https://drive.google.com/file/d/1nOo32l3yCJPamgxc3_bWO-pnsETS3uoh/view?usp=sharing) for the most recent working paper pdf.

## Replication

To replicate the figures, simply run this repository as a python package. For example:

```python
git clone https://github.com/alipatti/spectral-radius &&
    cd spectral-radius &&
    pip install . &&
    python -m spectral_radius
```

This will download the required data and generate everything necessary to compile the pdf:

```python
latexmk paper/main -pv
```


