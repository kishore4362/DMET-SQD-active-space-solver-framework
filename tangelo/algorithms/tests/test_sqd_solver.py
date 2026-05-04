# Copyright SandboxAQ 2021-2024.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from tangelo.algorithms.sqd_solver import SQDSolver


class SQDSolverRuntimeOptionsTest(unittest.TestCase):

    def test_runtime_sampler_options_support_error_mitigation(self):
        solver = SQDSolver({
            "molecule": object(),
            "backend": "ibm_torino",
            "service": object(),
            "dynamical_decoupling": {
                "enable": True,
                "sequence_type": "XY4",
                "scheduling_method": "alap",
            },
            "gate_twirling": True,
            "measurement_twirling": {
                "enable_measure": True,
                "num_randomizations": 16,
            },
        })

        self.assertEqual(
            solver._get_runtime_sampler_options(),
            {
                "dynamical_decoupling": {
                    "enable": True,
                    "sequence_type": "XY4",
                    "scheduling_method": "alap",
                },
                "twirling": {
                    "enable_gates": True,
                    "enable_measure": True,
                    "num_randomizations": 16,
                },
            },
        )

    def test_runtime_sampler_options_merge_with_raw_sampler_options(self):
        solver = SQDSolver({
            "molecule": object(),
            "backend": "ibm_torino",
            "service": object(),
            "sampler_options": {
                "twirling": {
                    "strategy": "active-accum",
                    "shots_per_randomization": 64,
                },
            },
            "dynamic_decoupling": True,
            "gate_twirling": {"enable_gates": True, "num_randomizations": 8},
        })

        self.assertEqual(
            solver._get_runtime_sampler_options(),
            {
                "dynamical_decoupling": {"enable": True},
                "twirling": {
                    "strategy": "active-accum",
                    "shots_per_randomization": 64,
                    "enable_gates": True,
                    "num_randomizations": 8,
                },
            },
        )

if __name__ == "__main__":
    unittest.main()
