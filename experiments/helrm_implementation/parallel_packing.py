import os
import sys
import torch
from helrm import HELRM

# check if the DLRM model path is provided
if len(sys.argv) != 2:
    print("Usage: python generete_weights.py <dlrm_model_path>")
    sys.exit(1)

# load model
model_pt = sys.argv[1]
state_dict  = torch.load(model_pt)

# print keys
for key in state_dict.keys():
    print(key, state_dict[key].shape)

# get the layer number from the key
layer_number = lambda string: int(string.split(".")[1])

# get the total vocab size from the state dict
def get_total_vocab_size(state_dict):
    vocab_sizes = []
    curr_embedding_table = 0
    curr_size = 0
    for key in state_dict.keys():
        if "emb_l" in key:
            if layer_number(key) == curr_embedding_table:
                curr_size += state_dict[key].shape[0]
            else:
                vocab_sizes.append(curr_size)
                curr_embedding_table = layer_number(key)
                curr_size = state_dict[key].shape[0]
    vocab_sizes.append(curr_size)
    return vocab_sizes

# instantiate an HELRM model
dense_size = state_dict["bot_l.0.weight"].shape[1]
vocab_sizes = get_total_vocab_size(state_dict)
helrm = HELRM(num_dense=dense_size, vocab_sizes=vocab_sizes, hidden_dim=2)

# copy weights from the state dict to the HELRM model
# we need to do this manually because of the differences 
# in the way the networks are defined
with torch.no_grad():
    # copy bot_l weights: always a fixed size so hardcoded
    # index is offset by 1 because of the extraction layer in Orion
    helrm.bot_l[1].weight[:] = state_dict["bot_l.0.weight"]
    helrm.bot_l[1].bias[:] = state_dict["bot_l.0.bias"]
    helrm.bot_l[3].weight[:] = state_dict["bot_l.2.weight"]
    helrm.bot_l[3].bias[:] = state_dict["bot_l.2.bias"]
 
    # copy top_l weights: again, fixed size so hardcoded
    helrm.top_l[0].weight[:] = state_dict["top_l.0.weight"]
    helrm.top_l[0].bias[:] = state_dict["top_l.0.bias"]
    helrm.top_l[2].weight[:] = state_dict["top_l.2.weight"]
    helrm.top_l[2].bias[:] = state_dict["top_l.2.bias"]
    helrm.top_l[4].weight[:] = state_dict["top_l.4.weight"]
    helrm.top_l[4].bias[:] = state_dict["top_l.4.bias"]

    # copy embedding tables
    list_of_embs = []
    curr_embedding_table = 0
    curr_embs = []
    for key in state_dict.keys():
        if "emb_l" in key:
            curr_layer_num = layer_number(key)
            if curr_layer_num != curr_embedding_table:
                list_of_embs.append(torch.cat(curr_embs, dim=0))
                curr_embs = []
                curr_embedding_table = curr_layer_num
            curr_embs.append(state_dict[key])


    list_of_embs.append(torch.cat(curr_embs, dim=0))
    # the following line packs embedding tables in a block-diagonal fashion
    helrm.embs[1].weight.data[:] = torch.block_diag(*list_of_embs).T


parent_path = os.path.dirname(model_pt)
sub_path = model_pt.split("/")[-1].split(".")[0]
torch.save(helrm.state_dict(), "helrm.pth"))

