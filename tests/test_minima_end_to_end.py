import pytest
import io
import math

from ase.io import read
from fairchem.core import FAIRChemCalculator
from fairchem.core.units.mlip_unit import load_predict_unit
from sella import Sella

F_MAX = 0.01

predictor = load_predict_unit(
        path='SM Conserving ALL.pt',
        device='cpu',
    )

calculator = FAIRChemCalculator(
        predictor,
        task_name='omol',
)

@pytest.mark.parametrize('input_traj, final_energy',
                         [('apalutamide.traj', -55272.146167),
                          ('lenvatinib.traj', -48759.030540)
                          ])

def test_minima_end_to_end(input_traj, final_energy):
    """
    Runs an end-to-end minima test with input atoms that have known final energies with set fmax.
    Known defined to be the same as the final energy found by the recent stable release of Sella,
    using OMOL25's eSEN-sm-conserving model as the calculator.

    Written by: Andrew Nguyen, amknguyen@berkeley.edu, 01/2026
    """
    atom = read(input_traj)

    atom.info = {"charge": 0, "spin": 1}
    atom.calc = calculator

    log_stream = io.StringIO()

    optimizer = Sella(
        atom,
        internal = True,
        trajectory = None,
        logfile = log_stream,
        order = 0
    )

    optimizer.run(fmax = F_MAX)

    log_content = log_stream.getvalue().splitlines()

    if log_content:
        converged_energy = float(log_content[-1].split(' ')[6]) #Reviewing the split, this is where the final energy is stored.
        assert math.isclose(converged_energy, final_energy, rel_tol=1e-9)
        print(f'converged energy is {converged_energy} and final energy is {final_energy}')
    else:
        raise Exception(f"{input_traj} did not converge! Cannot perform end-to-end minima test!")