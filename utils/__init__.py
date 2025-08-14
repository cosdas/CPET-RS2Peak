from pathlib import Path

import yaml

root_path = Path(__file__).resolve().parent.parent

code = yaml.load(open(Path(root_path / 'data/codebook.yml'), 'r'), Loader=yaml.FullLoader)
folds_path = Path(root_path / 'data/folds')
