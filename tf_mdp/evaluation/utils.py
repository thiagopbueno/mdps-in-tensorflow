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

import numpy as np


def initial_state(state, batch_size):
    return np.repeat([state], batch_size, axis=0).astype(np.float32)


def timesteps(batch_size, max_time):
    timesteps = [np.arange(start=max_time-1, stop=-1.0, step=-1.0, dtype=np.float32)]
    timesteps = np.repeat(timesteps, batch_size, axis=0)
    timesteps = np.reshape(timesteps, (batch_size, max_time, 1))
    return timesteps
