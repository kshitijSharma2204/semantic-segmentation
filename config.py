import importlib.util
import os

# repo root, so data/ and outputs/ resolve no matter where a script is run from
ROOT = os.path.dirname(os.path.abspath(__file__))


def load_labels(data_root):
    # The devkit's labels.py is a loose file inside data/, so I load it by path
    # instead of putting the repo on sys.path and importing data.devkit_semantics...
    path = os.path.join(data_root, "devkit_semantics/devkit/helpers/labels.py")
    spec = importlib.util.spec_from_file_location("kitti_devkit_labels", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.labels


def add_common_args(parser):
    parser.add_argument("--data-root", default=os.path.join(ROOT, "data"),
                        help="folder that holds data_semantics/ and devkit_semantics/")
    parser.add_argument("--out-root", default=os.path.join(ROOT, "outputs"),
                        help="where run output goes")
    parser.add_argument("--seed", type=int, default=42)
    return parser
