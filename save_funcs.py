import os
import pickle
import numpy as np
import jax

## SOURCE: https://github.com/deepmind/dm-haiku/issues/18#issuecomment-981814403
def save(ckpt_dir: str, state) -> None:
 with open(os.path.join(ckpt_dir, "arrays.npy"), "wb") as f:
   for x in jax.tree_util.tree_leaves(state):
     np.save(f, x, allow_pickle=False)

 tree_struct = jax.tree_map(lambda t: 0, state)
 with open(os.path.join(ckpt_dir, "tree.pkl"), "wb") as f:
   pickle.dump(tree_struct, f)

def restore(ckpt_dir):
 with open(os.path.join(ckpt_dir, "tree.pkl"), "rb") as f:
   tree_struct = pickle.load(f)
 
 leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
 with open(os.path.join(ckpt_dir, "arrays.npy"), "rb") as f:
   flat_state = [np.load(f) for _ in leaves]

 return jax.tree_util.tree_unflatten(treedef, flat_state)
 ###############################################################################