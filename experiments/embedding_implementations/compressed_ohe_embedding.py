import sys
import time
import numpy as np

import torch
import matplotlib.pyplot as plt

import orion
import orion.nn as on


torch.manual_seed(0)
orion.init_scheme("./resnet.yml")


try:
    p = int(sys.argv[1])
    l = int(sys.argv[2])
    hidden = int(sys.argv[3])
except:
    print("Usage: python compressed_ohe_embedding.py <p> <l> <hidden>")
    exit()


em = on.CompressedEmbedding(p, l, hidden) # https://arxiv.org/pdf/1711.01068
em.eval()

x = torch.randint(0, p, size=(l,))
inp = torch.nn.functional.one_hot(x, p).flatten().float()
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
times = []
for i in range(1):
    start = time.time()
    out_ctxt = em(vec_ctxt)
    # print(out_ctxt.scale())
    print(f"Runtime: {time.time() - start:.4f} secs.")
    times.append(time.time() - start)

# print(times)
# print(f"Average runtime: {sum(times) / len(times):.4f} secs.")
# print(f"STD: {np.std(times):.4f} secs.")
out_ptxt = out_ctxt.decrypt()
out_fhe = out_ptxt.decode().squeeze(0)

if hidden < 10:
    print(out_fhe)
    print(out_clear)
print("L2 error: ", torch.norm(out_fhe - out_clear))
