import torch
import numpy as np

def calc_RC_series_impedance(Rvalue: torch.Tensor, 
                                Value: torch.Tensor, 
                                omega: torch.Tensor) -> torch.Tensor:
    """Returns the impedance of an RC series circuit given the circuit parameters.

    Parameters
    ----------
    Rvalue : torch.Tensor
        Resistance in Ohm
    Value : torch.Tensor
        Capacitance in Farad
    omega : torch.Tensor
        Circular frequency in rad/s

    Returns
    -------
    torch.Tensor
        Impedance of RC series circuit.
    """
    return Rvalue - 1.j / (omega * Value)


def calc_RL_series_impedance(Rvalue: torch.Tensor, 
                                Value: torch.Tensor, 
                                omega: torch.Tensor) -> torch.Tensor:
    """Returns the impedance of an RL series circuit given the circuit parameters.

    Parameters
    ----------
    Rvalue : torch.Tensor
        Resistance in Ohm
    Value : torch.Tensor
        Inductance in Henry
    omega : torch.Tensor
        Circular frequency in rad/s

    Returns
    -------
    torch.Tensor
        Impedance of RL series circuit.
    """
    return Rvalue + 1.j * omega * Value


def calc_RC_parallel_impedance(Rvalue: torch.Tensor, 
                                Value: torch.Tensor, 
                                omega: torch.Tensor) -> torch.Tensor:
    """Returns the impedance of an RC parallel circuit given the circuit parameters.

    Parameters
    ----------
    Rvalue : torch.Tensor
        Resistance in Ohm
    Value : torch.Tensor
        Capacitance in Farad
    omega : torch.Tensor
        Circular frequency in rad/s

    Returns
    -------
    torch.Tensor
        Impedance of RC parallel circuit.
    """
    return Rvalue*(1. - 1j*omega*Rvalue*Value) / ((omega*Rvalue*Value)**2 + 1)


def calc_RL_parallel_impedance(Rvalue: torch.Tensor, 
                                Value: torch.Tensor,
                                omega: torch.Tensor) -> torch.Tensor:
    """Returns the impedance of an RL parallel circuit given the circuit parameters.

    Parameters
    ----------
    Rvalue : torch.Tensor
        Resistance in Ohm
    Value : torch.Tensor
        Inductance in Henry
    omega : torch.Tensor
        Circular frequency in rad/s

    Returns
    -------
    torch.Tensor
        Impedance of RL parallel circuit.
    """
    return Rvalue*(1. - 1j*omega*(Value/Rvalue)) / ((omega*(Value/Rvalue))**2 + 1)


def z_to_s(z: torch.Tensor, 
            z0: torch.Tensor) -> torch.Tensor:
    """Converts impedance to scattering parameters.

    Parameters
    ----------
    z : torch.Tensor
        real-valued 1d-tensor
    z0 : torch.Tensor
        real-valued 1d-tensor

    Returns
    -------
    torch.Tensor
        scattering parameters
    """
    return (z - z0) / (z + z0)

def reshape_1_port_networks(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.permute(0, -1)
    tensor = tensor.unsqueeze(1)

def extract_initial_params(omega, y):
    """Extracts initial parameters from Y-Admittance matrix
    Essentially we calculate the circuit parameters of each individual circuit element that would theoretically make the structure resonant.
    This was only tested on split-rings and may not be applicable to other structures.
    Input: torch.tensor of shape [n0, n0] - Admittance has to be passed without inclusion of excitation
    Output: torch.tensor of shape [n0] - circuit parameters that would make the structure resonant
    """
    L = -1/(omega*torch.sum(torch.imag(y), dim=0))
    C = 1/(omega**2*L)
    return C

def export_params_to_txt(params: np.ndarray, 
                         variable: str, 
                         filename: str) -> None:
    """Exports params to txt file primarily for use in CST
    """
    param_list = [f'{variable}{i+1}="{val:.4f}"' for i, val in enumerate(params)]
    with open(filename, "w") as file:
        for param in param_list:
            file.write(param + '\n')