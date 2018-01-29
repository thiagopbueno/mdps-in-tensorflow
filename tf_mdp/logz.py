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
import numpy as np
import os
import time


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


def serialize(file_path, data):
    with open(file_path, "w") as file:
        data = json.dumps(data, cls=NumpyEncoder, sort_keys=True, indent=4)
        file.write(data)


def deserialize(file_path):
    with open(file_path, "r") as file:
        data = json.loads(file.read())
        return data


def logging(log_dir, data):
    try:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        file_path = os.path.join(log_dir, timestamp + ".json")
        print(">> " + file_path)
        serialize(file_path, data)

        return file_path

    except IOError:
        print("I/O error", file_path)
