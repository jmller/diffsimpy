#import numpy as np
import math
import torch
import os
import sys
import pytest
import numpy as np

packaging = False # Set to False to run tests in the developement stage

if not packaging:
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
    sys.path.append(src_path)
from DiffSimPy import DiffCoSimulator

batch_size = 32 # Batch size
nf = 101 # Num of frequencies
n0 = 6 # Num of ports when not connected to networks
n1 = 2 # Num of ports when connected to networks
npoints= 512 # Num of points in the field

torch.manual_seed(42)
S_0 = torch.rand((batch_size, nf, n0, n0), dtype=torch.cdouble)
S_C = torch.rand((batch_size, nf, n0+n1, n0+n1), dtype=torch.cdouble)

def test_calc_voltage_wave():
    ds = DiffCoSimulator(S_0 = S_0, S_C_const = S_C)
    V_O_p = ds.calc_voltage_wave(S_C)
    assert V_O_p.shape == (batch_size, nf, n0, n1)
    
def test_calc_Pinc():
    ds = DiffCoSimulator()
    V_O_p = torch.rand((batch_size, nf, n0, n1), dtype=torch.cdouble)
    p_incM, phaseM = ds.calc_Pinc(V_O_p)
    assert p_incM.shape == (batch_size, nf, n0, n1)
    assert phaseM.shape == (batch_size, nf, n0, n1)

def test_calc_norm():
    ds = DiffCoSimulator()
    p_incM = torch.rand((batch_size, nf, n0, n1), dtype=torch.cdouble)
    phaseM = torch.rand((batch_size, nf, n0, n1), dtype=torch.cdouble)
    norm = ds.calc_norm(p_incM, phaseM)
    assert norm.shape == (batch_size, nf, n0, n1)

def test_transform_fields_with_norm():
    ds = DiffCoSimulator()
    norm = torch.rand((batch_size, nf, n0, n1), dtype=torch.cdouble)
    field = torch.rand((nf, n0, 3, npoints), dtype=torch.cdouble)
    field_transformed = ds.transform_fields_with_norm(field, norm)
    assert field_transformed.shape == (batch_size, nf, n1, 3, npoints)