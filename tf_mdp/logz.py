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

import csv
import os
import time


def logging(log_dir, data):
    try:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        csv_file = os.path.join(log_dir, timestamp + ".dat")
        print(">> " + csv_file)

        with open(csv_file, 'w') as csvfile:
            writer = csv.writer(csvfile)

            # header
            writer.writerow(sorted(data.keys()))

            # content
            values = [data[key] for key in sorted(data)]
            for row in zip(*values):
                writer.writerow(row)

    except IOError:
        print("I/O error", csv_file)
