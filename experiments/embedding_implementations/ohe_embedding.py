import sys
import time

import torch
import matplotlib.pyplot as plt

import orion
import orion.nn as on


torch.manual_seed(0)
orion.init_scheme("./resnet.yml")

vocab_size = 8
hidden = 5

em = on.Embedding(vocab_size, hidden) # simple linear layer impl
em.eval()

x = torch.randint(0, vocab_size, size=(1,))
inp = torch.nn.functional.one_hot(x, vocab_size).flatten().float()
inp = inp[None,:]

with torch.no_grad():
    out_clear = em(inp).squeeze(0)
    
# Fit and compile
orion.fit(em, inp)
input_level = orion.compile(em)
print(input_level)

# Encode and encrypt
vec_ptxt = orion.encode(inp)
vec_ctxt = orion.encrypt(vec_ptxt)

em.he()

print("\nStarting FHE inference")
start = time.time()
out_ctxt = em(vec_ctxt)
print(out_ctxt.scale())
print(f"Runtime: {time.time() - start:.4f} secs.")

out_ptxt = out_ctxt.decrypt()
out_fhe = out_ptxt.decode().squeeze(0)

print(out_fhe)
print(out_clear)
print("L2 error: ", torch.norm(out_fhe - out_clear))

