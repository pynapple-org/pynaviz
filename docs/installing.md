# Installation

We recommend using the **Qt-based interface** for the best interactive experience:



```bash
git clone https://github.com/pynapple-org/pynaviz.git
cd pynaviz
pip install -e '.[qt]'
```

If Qt is not available on your system, you can still use the fallback rendering engine (via PyGFX):

```bash
git clone https://github.com/pynapple-org/pynaviz.git
cd pynaviz
pip install -e .
```