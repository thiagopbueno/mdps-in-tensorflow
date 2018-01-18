# This file is part of TF-MDP.

# TF-MDP is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# TF-MDP is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with TF-MDP.  If not, see <http://www.gnu.org/licenses/>.

import json
import importlib
import os

PATH = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(PATH, "models.json")

import pkgutil
# print([name for _, name, _ in pkgutil.iter_modules(['.'])])
# print([name for _, name, _ in pkgutil.iter_modules([PATH])])

def make(problem_id):
    with open(MODELS, 'r') as file:
        metadata = json.loads(file.read())[problem_id]
        module = metadata["module"]
        class_name = metadata["class_name"]
        config = metadata["config"]
        # print(module, class_name, config)
        m = importlib.import_module("models.{}".format(module))
        model = getattr(m, class_name)
        return model, config
