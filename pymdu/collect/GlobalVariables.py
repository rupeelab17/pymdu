# ******************************************************************************
#  This file is part of pymdu.                                                 *
#                                                                              *
#  Copyright                                                                   *
#                                                                              *
#  pymdu is free software: you can redistribute it and/or modify               *
#  it under the terms of the GNU General Public License as published by        *
#  the Free Software Foundation, either version 3 of the License, or           *
#  (at your option) any later version.                                         *
#                                                                              *
#  pymdu is distributed in the hope that it will be useful,                    *
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              *
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               *
#  GNU General Public License for more details.                                *
#                                                                              *
#  You should have received a copy of the GNU General Public License           *
#  along with pymdu.  If not, see <https://www.gnu.org/licenses/>.             *
# ******************************************************************************
import os
import tempfile
from pathlib import Path

# BBOX_LR = [-1.158805, 46.142985, -1.152736, 46.149185]
# Villeneuve les Salines
# BBOX_LR = [-1.1344835551543042, 46.148883371762906, -1.1156389607645565, 46.15501967092246]
BBOX_LR = [-1.152704, 46.181627, -1.139893, 46.186990]
MAIN_PATH = os.path.dirname(os.path.dirname(Path(__file__)))
MY_LOCAL_PATH = MAIN_PATH + '/example/'
try:
    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     temp_dir = Path(tmpdirname)
    #     print(temp_dir, temp_dir.exists())

    TEMP_PATH = tempfile.gettempdir()
    # TEMP_PATH = './'
    print('TEMP_PATH', tempfile.gettempdir())

except Exception:
    TEMP_PATH = MAIN_PATH + '/example/'
