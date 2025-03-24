# INTEGRATIONTEST
# BCC Test Case for DiffSimPy
import numpy as np
import torch
import os
import sys
# use locally installed cosimpy
#import sys, importlib.util; sys.path.insert(0, '/workspace/code/cosimpy/src'); cosimpy_spec = importlib.util.find_spec('cosimpy'); cosimpy = importlib.util.module_from_spec(cosimpy_spec); cosimpy_spec.loader.exec_module(cosimpy)
from cosimpy import S_Matrix, EM_Field, RF_Coil #TODO
import matplotlib.pyplot as plt

packaging = False # Set to False to run tests in the developement stage

if not packaging:
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    sys.path.append(src_path)
from DiffSimPyPipeline import *

# Global Parameters
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
path = r"testdata/BCC"
freq = 123.5
numPorts = 2
nPoints=[21, 5, 11]

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

s_matrix_unconnected = torch.from_numpy(s_matrix.S).to(torch.cdouble)
b_field_import = torch.tensor(em_field.b_field, dtype=torch.cdouble)
e_field_import = torch.tensor(em_field.e_field, dtype=torch.cdouble)
f_idx = np.where(s_matrix.frequencies == 123.5e6)[0][0]


# Build Network Matrix including 2 Port tuning network
R_tune = 40 #Ohm
C_tune = 17.8e-12 #F

# Init Cosimpy Reference
rf_coil = RF_Coil(s_matrix=s_matrix, em_field=em_field)
tuning_circuits = [
    S_Matrix.sMatrixPInetwork(None, None, S_Matrix.sMatrixRCseries(R_tune, C_tune, freqs=rf_coil.s_matrix.frequencies, z0=50)),
    S_Matrix.sMatrixPInetwork(None, None, S_Matrix.sMatrixRCseries(R_tune, C_tune, freqs=rf_coil.s_matrix.frequencies, z0=50))
]
rf_coil_conn = rf_coil.singlePortConnRFcoil(networks = tuning_circuits, comp_Pinc=True)
b_field_ref = rf_coil_conn.em_field.b_field[0] #because the b_field only exists for one frequency
e_field_ref = rf_coil_conn.em_field.e_field[0]


# Init DiffSimPy
networks = torch.tensor([tuning_circuits[0].S,
                        tuning_circuits[1].S], dtype=torch.cdouble)

# n0 = n1 = 2
# This approach has the advantage that any network matrix can be used during simulation
S_C_22 = torch.diag(networks[:, f_idx, 1, 1]).unsqueeze(0).unsqueeze(0)
S_C_11 = torch.diag(networks[:, f_idx, 0, 0]).unsqueeze(0).unsqueeze(0)
S_C_12 = torch.diag(networks[:, f_idx, 0, 1]).unsqueeze(0).unsqueeze(0)
S_C_21 = torch.diag(networks[:, f_idx, 1, 0]).unsqueeze(0).unsqueeze(0)
S_C = torch.cat((torch.cat((S_C_11, S_C_21), dim=-1), 
                         torch.cat((S_C_12, S_C_22), dim=-1)), dim=-2) 

indices = None
calc_pipe = CalculationPipeline(s_matrix_unconnected, 
                                S_C,
                                b_field_import,
                                e_field_import)
diffsimpy_out = calc_pipe(None, None)

assert torch.allclose(diffsimpy_out[0][0, f_idx, :, :, :], torch.tensor(b_field_ref, dtype=torch.cdouble), atol=1e-6)
assert torch.allclose(diffsimpy_out[1][0, f_idx, :, :, :], torch.tensor(e_field_ref, dtype=torch.cdouble), atol=1e-6)