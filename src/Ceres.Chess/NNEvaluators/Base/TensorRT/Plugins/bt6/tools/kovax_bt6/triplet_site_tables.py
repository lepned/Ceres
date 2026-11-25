"""Table blob of the BT6TripletSite plugin (bt6_triplet_site.cu) from a BT6 carrier's tensors (numpy only).

Layout, FP32, every map [out][in] exactly as the carrier stores it (bt6_net.served_site's einsum "oc,bc..."):
  offset    0  R    [16][heads]  /encoder{i}/edge_site/readback/w          (heads = the layer's attention heads, 32 here)
  +16 h        Wv   [32][16]     /encoder{i}/edge_site/triplet/value/w     row = direction * 16 + head * 4 + dot
  +512         Wg   [16][16]     /encoder{i}/edge_site/triplet/gate/w      row = e_in(h 0..3) | g_in | e_out | g_out
  +256         bg   [16]         /encoder{i}/edge_site/triplet/gate/b
  +16          Wo   [16][32]     /encoder{i}/edge_site/triplet/out/w       column = va channel (direction * 16 + h * 4 + d)
  +512         W1   [64][16]     /encoder{i}/edge_site/ffn/dense1/w
  +1024        Wr   [64][16]     /encoder{i}/edge_site/ffn/dense1_rev/w    (zeros when the net has no reverse edge)
  +1024        b1   [64]         /encoder{i}/edge_site/ffn/dense1/b
  +64          W2   [16][64]     /encoder{i}/edge_site/ffn/dense2/w
  total = 16 * heads + 4432 floats (4944 for 32 heads, 4816 for 24).

    tab = triplet_site_table(tensors, site_layer)      # tensors: {carrier name: array}, e.g. bt6_carrier.read_carrier
"""
import numpy as np

SITE_TAB_FIXED = 512 + 256 + 16 + 512 + 1024 + 1024 + 64 + 1024


def site_tab_size(heads: int) -> int:
  return 16 * heads + SITE_TAB_FIXED


def triplet_site_table(tensors, layer: int) -> np.ndarray:
  s = f"/encoder{layer}/edge_site"
  f = lambda name: np.asarray(tensors[f"{s}/{name}"], dtype=np.float32)
  R = f("readback/w")
  heads = R.shape[1]
  parts = [(R, (16, heads)), (f("triplet/value/w"), (32, 16)), (f("triplet/gate/w"), (16, 16)),
           (f("triplet/gate/b"), (16,)), (f("triplet/out/w"), (16, 32)), (f("ffn/dense1/w"), (64, 16)),
           (f("ffn/dense1_rev/w") if f"{s}/ffn/dense1_rev/w" in tensors else np.zeros((64, 16), np.float32), (64, 16)),
           (f("ffn/dense1/b"), (64,)), (f("ffn/dense2/w"), (16, 64))]
  for array, shape in parts:
    assert array.shape == shape, (layer, array.shape, shape)
  tab = np.concatenate([a.ravel() for a, _ in parts]).astype(np.float32)
  assert tab.size == site_tab_size(heads)
  return tab
