# MLP – Breast Cancer

## Installation

Clone the project, enter the project directory, then run:

```bash
./init.sh
```

This installs the required dependencies.

## Mandatory

The mandatory part is implemented with:

* `mlp.py`
* `mlp_core.py`
* `utils.py`

Run the MLP with:

```bash
python mlp.py
```

The dataset used is `data.csv`.

## Bonus

Two bonus implementations are provided:

* **Adam optimizer**

  * `adammlp.py`
  * `adammlp_core.py`
  * `adamutils.py`

* **Adam + Early Stopping**

  * `adammlp_es.py`
  * `adammlp_core_es.py`
  * `adamutils_es.py`

The project also includes the corresponding training and validation workflow.
