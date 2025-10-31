import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lime.regression import regression

reg = regression(
    batch_size=5,
    n_segments=50,
    compactness= 10,
    target_class=8,
    alpha=0.001
)
reg.visualiser_top_superpixels(out_dir="/home/jmerhrioui/soprasteria/poc/test/top_results") # attentioon ckpt à changer dans fct + data load
# l.110 - 117

