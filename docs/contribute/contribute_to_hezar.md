# Contribute to Hezar

1. Clone the repository to your local machine:
```bash
 git clone https://github.com/hezarai/hezar.git
 cd hezar
```

2. Create a virtual environment with `python>=3.8`, activate it, install the required
   dependencies and install the pre-commit configuration:

```bash
conda create -n hezar_env python
conda activate hezar_env
pip install -r requirements.txt
pre-commit install
```

3. Create a branch and commit your changes:
```bash
git switch -c <name-your-branch>
# do your changes
git add .
git commit -m "your commit msg"
git push
```

4. Merge request to `main` for review.
