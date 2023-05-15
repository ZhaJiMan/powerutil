import shutil
from pathlib import Path

from powerutils.common import new_dir

dirpath_root = Path(__file__).parent.parent
dirpath_data = dirpath_root / 'data'
dirpath_model = dirpath_root / 'model'
dirpath_fig = dirpath_root / 'fig'
new_dir(dirpath_data)
new_dir(dirpath_model)
new_dir(dirpath_fig)

dirpath_merge = dirpath_data / 'merge'
new_dir(dirpath_merge)

name = ''
cap = 100
lon = 100
lat = 30