# HE-LRM: Efficient Private Embedding Lookups for Neural Inference Using Fully Homomorphic Encryption

This anonymous repository contains the code to run the HE-LRM architecture under FHE.
```
.
├── experiments
│   ├── embedding_implementations
│   ├── helrm_implementation
│   └── moai_cp_mm
└── orion
```

## Experiments
### Embedding Implementations
```bash
cd experiments/embedding_implementations
```
The `embedding_implementations` directory has both our compressed embedding implementation and the baseline we use in the paper ([https://openreview.net/pdf?id=apxON2uH4N](https://openreview.net/pdf?id=apxON2uH4N)).
You can run the baseline:
```
python icml_baseline_embedding.py 8 8 50
```
where the arguments are `p`, `l`, and `hidden_dim`, respectively.
Run our implementation with the same parameters:
```
python compressed_ohe_embedding.py 8 8 50
```
### HELRM
```bash
cd experiments/helrm_implementation
```
The `helrm_implementation` directory contains both a cleartext DLRM model as well as the HELRM equivalent model. Our HELRM model is at `helrm.py` and our parallel embedding packing strategy is at `parallel_packing.py`.

Run the PyTorch DLRM model:
```
python dlrm.py
```
Run an equivalent and FHE-friendly HELRM model in the clear:
```
python helrm.py
```
Finally, run this HELRM architecture under FHE:
```
python run_helrm.py
```

### MOAI's CP-MM Algorithm
```bash
cd experiments/moai_cp_mm
```

Compile and run for various sequence lengths, hidden dimensions and vocabulary size.
```bash
go build -o scalar_mult_example .
./scalar_mult_example --seq_len=5 --d=10 --V=12 --runs=1
```

## Installation
We tested our implementation on `Ubuntu 22.04.5 LTS`. First, install the required dependencies:

```
sudo apt update && sudo apt install -y \
    build-essential git wget curl ca-certificates \
    python3 python3-pip python3-venv \
    unzip pkg-config libgmp-dev libssl-dev
```

Install go (for installing Orion):

```
cd /tmp
wget https://go.dev/dl/go1.22.3.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.22.3.linux-amd64.tar.gz
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
cd ~/emb
go version # go version go1.22.3 linux/amd64
```

### Install Orion
```
cd orion
pip install -e .
```

We modified or added the following Orion files to add Embedding tables:
```
orion/orion/nn/__init__.py
orion/orion/nn/activation.py
orion/orion/nn/embedding.py
orion/orion/nn/extract.py
orion/orion/nn/module.py
orion/orion/backend/python/encoder.py
orion/orion/backend/python/encryptor.py
orion/orion/backend/python/poly_evaluator.py
orion/orion/backend/python/tensors.py
```
