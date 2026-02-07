import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

import orion
import orion.nn as on


torch.manual_seed(0)
orion.init_scheme("./resnet.yml")


# p -> (r, iter), table 6 in https://openreview.net/pdf?id=apxON2uH4N
mapping = {4: (6, 1), 
           8: (8, 1),
           16: (10, 2),
           32: (11, 2),
           64: (13, 3)}

try:
    p = int(sys.argv[1])
    l = int(sys.argv[2])
    hidden = int(sys.argv[3])
except:
    print("Usage: python icml_baseline_embedding.py <p> <l> <hidden>")
    exit()

r, iter = mapping[p]

em = on.BaselineEmbedding(p, l, r, iter, hidden) # https://openreview.net/pdf?id=apxON2uH4N
em.eval()

x = torch.randint(0, p, size=(l,))
x = torch.repeat_interleave(x, p).float()
inp = x - torch.arange(p).repeat(l)

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
