# INTEGRATIONTEST
# Simple Test-Case (3D)
import numpy as np
import torch
import os
import sys
#import sys, importlib.util; sys.path.insert(0, '/workspace/code/cosimpy/src'); cosimpy_spec = importlib.util.find_spec('cosimpy'); cosimpy = importlib.util.module_from_spec(cosimpy_spec); cosimpy_spec.loader.exec_module(cosimpy)
from cosimpy import S_Matrix, EM_Field, RF_Coil #TODO
import matplotlib.pyplot as plt

packaging = False # Set to False to run tests in the developement stage

if not packaging:
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    sys.path.append(src_path)
from DiffSimPyPipeline import *
import NetworkFunctions

# Global Parameters
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
path = r"testdata/3x1_volume"
freq = 123.5
numPorts = 4
nPoints=[39, 14, 9]

# Import of Data
s_matrix = S_Matrix.importTouchstone(os.path.join(THIS_DIR, path, f"s_param.s{numPorts}p"))
em_field = EM_Field.importFields_cst(os.path.join(THIS_DIR, path, "Field"), 
                                        freqs=[freq], 
                                        nPorts=numPorts, 
                                        Pinc_ref=1, 
                                        b_multCoeff=1,
                                        pkORrms='rms',
                                        fileType = 'ascii', 
                                        col_ascii_order = 1)
f_idx = np.where(s_matrix.frequencies == 123.5e6)[0][0]

s_matrix_unconnected = torch.from_numpy(s_matrix.S).to(torch.cdouble)
b_field_import = torch.tensor(em_field.b_field, dtype=torch.cdouble)
e_field_import = torch.tensor(em_field.e_field, dtype=torch.cdouble)

# Calculate statevectors
np.random.seed(42)
Rs_list = np.random.rand(10, 3)*10
Cs_list = np.random.rand(10, 3)*10e-12

# Init CoSimPy Calculation Pipeline
rf_coil = RF_Coil(s_matrix=s_matrix, em_field=em_field)

# Init DiffSimPy Calculation Pipeline
S_C = torch.tensor([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0]], dtype=torch.cdouble)

calc_pipe = CalculationPipeline(s_matrix_unconnected, 
                                S_C,
                                b_field_import,
                                e_field_import)

indices = torch.tensor([[0, 1, 2],
           [0, 1, 2]], dtype=torch.int)
omega = torch.tensor([2*torch.pi*freq*10**6], dtype=torch.cdouble)

# Calc CoSimPy Reference
ref_fields = []
for Rs, Cs in zip(Rs_list, Cs_list):
    RC_circuits = [S_Matrix.sMatrixRCseries(Rs[x], Cs[x], freqs=s_matrix.frequencies, z0=50) for x in range(numPorts-1)]    
    networks = RC_circuits + [None]
    rf_coil_conn = rf_coil.singlePortConnRFcoil(networks = networks, comp_Pinc=True)
    b_field_ref = rf_coil_conn.em_field.b_field[0]
    e_field_ref = rf_coil_conn.em_field.e_field[0]
    ref_fields.append([b_field_ref, e_field_ref])

z_networks = NetworkFunctions.calc_RC_series_impedance(torch.tensor(Rs_list), torch.tensor(Cs_list), torch.tensor([[omega]]))
tunable_s_params = NetworkFunctions.z_to_s(z_networks, 50)
tunable_s_params = tunable_s_params.permute(0, -1)
tunable_s_params = tunable_s_params.unsqueeze(1)
diffsimpy_out = calc_pipe(tunable_s_params, indices)

assert torch.allclose(diffsimpy_out[0][:, f_idx, :, :, :], torch.tensor(ref_fields, dtype=torch.cdouble)[:, 0], atol=1e-6)
assert torch.allclose(diffsimpy_out[1][:, f_idx, :, :, :], torch.tensor(ref_fields, dtype=torch.cdouble)[:, 1], atol=1e-6)