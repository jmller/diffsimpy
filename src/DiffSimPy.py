"""
Differentiable Co-Simulation Framework for the Efficient Modelling and Optimization of Electromagnetic Circuit Cosimulations.

**Motivation**
- Reconfigurable lumped elements of resonant structures significantly impact resonant behavior.
- Full-wave simulations are expensive and time-consuming.
- Electromagnetic Co-simulation provides an efficient method to determine the altered field distribution due to the tuning of lumped elements.
- Furthermore setting lumped elements to specific values is often a non-trivial task.
- The use of gradients could improve efficiency of the optimization in tunable parameters in resonant structures.

**Main Features**
The framework, built on PyTorch as the underlying matrix manipulation framework, offers the following features:
- **Parallel processing** of resulting electromagnetic fields from any given scattering parameters (S_C) and earlier preprocessing steps.
- Support for **gradient-based optimization** algorithms.
- GPU-accelerated computations.
- **Easy integration** into existing optimization pipelines.
- Compatibility with any circuit configuration, provided the scattering parameters are given.
- Full **control and reusability** due to the modular design.
- Adherence to the **standards of the CoSimPy Framework**, facilitating seamless integration into existing projects.

**Philosophy**
- The framework is designed to provide maximum flexibility to users:
    - Users should view the framework as a modular toolbox, usable in conjunction with frameworks like CoSimPy or scikit-rf.
    - Users only need to define the scattering parameters for the circuitry connected to the resonant structure.
    - Users can then leverage the default `CalculationPipeline` and pass scattering parameters to the `forward` method (see examples).
        - A custom preprocessing function containing all necessary preprocessing steps can be beneficial, depending on the optimization goal:
            - Some users may choose to optimize circuit parameters like capacitances, while others might focus on directly manipulating and optimizing the scattering parameters.
            - The custom preprocessing function or the `forward` method in the `CalculationPipeline` can be adjusted accordingly.
    - Two matrices, `S_C_const` and `S_C_var`, are used:
        - `S_C_const` is defined once during the instantiation of the class and remains fixed throughout the optimization process.
        - `S_C_var` is the tunable part of the scattering parameters, modified during the optimization process.
            - Users must define the indices of the scattering parameters to be optimized within the `S_C_var` matrix.
        - The final scattering matrix `S_C` is the sum of `S_C_const` and `S_C_var`.
            - To avoid calculation errors, elements defined in `S_C_const` should be set to 0 in `S_C_var`, and vice versa.
- The framework can also be used as a standalone tool for parallelized electromagnetic field calculations by utilizing only a static `S_C_const` matrix.
"""

import torch
from typing import Union

class DiffCoSimulator(torch.nn.Module):
    def __init__(self, 
                 S_0: torch.Tensor = None,
                 S_C_const: torch.Tensor = None, 
                 b_field: Union[torch.Tensor, None] = None, 
                 e_field: Union[torch.Tensor, None] = None,
                 z0: Union[torch.Tensor, float] = 50, #Ohm
                 P_inc: Union[torch.Tensor, float]= 1, #W
                 **kwarg):
        """Main Class of the Differentiable Co-Simulation Framework. 
        
        Dimensions of the tensors:
        ----------
        batch_size : int
            Number of samples in the batch that are being processed in parallel
        n0 : int
            Number of ports when not connected to networks i.e. number of ports of the unconnected structure
        n1 : int
            Number of ports when connected to networks i.e. number of ports of the connected structure. Those are the ports that are directly or indirectly exposed to an external excitation.
        nf : int
            Number of frequency points
        npoints : int
            Number of points in the field

        Parameters
        ----------
        S_0 : torch.Tensor
            2d-tensor of shape [n0, n0]; S-Matrix of simulated resonant structure from em-simulation (unconnected structure); n_0 is the number of ports.
        S_C_const : torch.Tensor
            4d-tensor of shape [batch_size, nf, n0+n1, n0+n1]; S-Matrix of the circuitry to be connected to the resonant structure, Explanation follows in Notes. Follows standards defined by CoSimPy.
        freqs : torch.Tensor
            1d-tensor of shape [nf]; Frequencies at which the simulation is performed
        z0 : torch.Tensor
            1d-tensor of shape [n0]; Characteristic impedance of the individual networks
        b_field or e_field : Union[torch.Tensor, None], optional
            4d-tensor of shape [nf, n0, 3, npoints]; Magnetic or electric fields of individual ports, by default None
        P_inc : torch.Tensor
            1d-tensor of shape [n0]; Incident power at each of the ports of the unconnected structure, by default 1W for all networks
        z0 : Union[torch.Tensor, float], optional
            Characteristic impedance of each of the networks, by default 50 Ohm for all networks
        Returns
        -------
        None

        Notes
        -----
        The Differential Co-Simulation Framework is designed to efficiently model the co-simulation of electromagnetic and circuit simulations.
        The workflow works as follows:
        The User instantiates a class object of the Differential Co-Simulation Framework that will be called repeatedly in an optimization loop.
        During instantiation the user needs to provide ...
        - the s-matrix (S0) of the unconnected resonant structure. The s-matrix can be loaded using frameworks like CoSimPy or scikit-rf.
        - S_C describes an intermediate Multiport network that handles the connection between the resonant structure (with its output_ports (n0)) and the resulting output_ports of the connected structure (n1). 
        The s-matrix (S_C) will be connected to the resonant structure. In general, S_C can be divided into 4 parts (S_C_11, S_C_12, S_C_21, S_C_22) where S_C_11 and S_C_22 are the reflection parameters and S_C_12 and S_C_21 are the transmission parameters. The S-Parameters of S_C_const will NOT be changed during optimization. 
        Parameters that will be changed during optimization will be passed to the Optimization pipeline as S_C_var. S_C was defined in line with the CoSimPy Framework.
        - e_field and b_field are the electric and magnetic fields of the individual ports. They are optional and can be passed to the framework. If passed to the framework, they will be used to calculate the electromagnetic fields. The passed fields are expected to be in the shape (nf, n0, 3, npoints). The CoSimPy field import method is fulfilling this criteria.
        - The characteristic impedance z0 of the networks can be passed as a tensor or a float. If a scalar float is passed, the same impedance will be used for all networks.
        - The incident power P_inc at each of the ports of the unconnected structure can be passed as a tensor or a float. If a scalar float is passed, the same incident power will be used for all networks.
        """
        super(DiffCoSimulator, self).__init__()
        self.default_real_dtype = torch.float64
        torch.set_default_dtype(self.default_real_dtype)
        if self.default_real_dtype == torch.float64:
            self.default_complex_dtype = torch.cdouble
        else:
            self.default_complex_dtype = torch.cfloat

        self.S_0 = S_0
        self.S_C_const = S_C_const
        self.z0 = z0
        self.P_inc = P_inc

        self.b_field = b_field if b_field is not None else None
        self.e_field = e_field if e_field is not None else None

    def assign_S_C_var(self,
                       s_params: torch.Tensor, #flat tensor of scattering parameters
                       sc_idx: torch.Tensor,
                       sc_idy: torch.Tensor) -> torch.Tensor:
        """Assigns the scattering parameters to the S_C_var matrix. In S_C_var the optimizable scattering parameters are stored in the shape (batch_size, n_f, n0+n1, n0+n1).

        Parameters
        ----------
        s_params : torch.Tensor
            flattened tensor of scattering parameters that wlil be assigned to the S_C matrix by the indices sc_idx and sc_idy
        sc_idx and sc_idy: torch.Tensor
            indices of the scattering parameters that are required for assignement of individual s_params elements to the S_C matrix. The length of sc_idx and sc_idy has to be equal to the number of scattering parameters in s_params.
        Returns
        -------
        S_C_var : torch.Tensor
            4d-tensor of shape [batch_size, n_f, n0+n1, n0+n1]
            The scattering matrix of the circuitry with the assigned scattering parameters. In upcoming processes, the scattering parameters of S_C_var will be added to S_C_const.
        """
        n0 = self.S_0.shape[-1]
        n1 = self.S_C_const.shape[-1] - n0

        nf = s_params.shape[1]
        ns = s_params.shape[-1]
        batch_size = s_params.shape[0]

        S_C = torch.zeros(batch_size, nf, (n0+n1), (n0+n1), dtype=self.default_complex_dtype).to(self.S_0.device)
        s_params = s_params.to(self.S_0.device)
        
        assert len(sc_idx) == len(sc_idy) == ns
        id = torch.arange(ns, dtype=torch.int)
        S_C[:, :, sc_idx[id], sc_idy[id]] = s_params[:, :, id]
        return S_C

    def build_S_C(self,
                  S_C_var: torch.Tensor) -> torch.Tensor:
        """Builds the S_C matrix from the scattering parameters of the networks by adding equally sized torch tensors.

        Parameters
        ----------
        S_C_var : torch.Tensor
            The scattering parameters of optimizable networks. For further information see assign_S_C_var method.

        Returns
        -------
        torch.Tensor
            3D tensor of shape [batch_size, nf, n0+n1, n0+n1]
            Combination of optimizable and constant scattering parameters of the networks.

        Notes
        -----
        The S_C matrix is constructed based on the scattering parameters of the networks.
        It has dimensions [batch_size, nf, n0+n1, n0+n1],

        For instance, RLC elements are modeled as passive 1-port networks and their reflection parameters
        are assigned to the diagonal elements of the matrix (e.g., `S_C[:, in_idx, in_idx]`).

        For excitation elements, a value of 1 is assigned for complete transmission/power when out of diagonal
        at the corresponding element.

        S_C is constructed such that:
        - `S_C_11` represents the reflection S-parameters from the input perspective.
        - `S_C_22` represents the reflection S-parameters from the output perspective.
        - `S_C_12` and `S_C_21` represent the transmission S-parameters from input to output and vice-versa.
        
        Example:
        ----------
        For a 1-port RLC element (passive network):
        - Reflection is assigned to diagonal elements: `S_C[:, :, in_idx, in_idx]`.
        
        For a 2-port network:
        - Transmission and reflection parameters are assigned accordingly:
        - `S_C_11`: Reflection from input (resonant structure perspective).
        - `S_C_22`: Reflection from output (connected ciruitry perspective).
        - `S_C_12` and `S_C_21`: Transmission of connected circuitry.

        """
        return self.S_C_const + S_C_var

    def calc_voltage_wave(self, S_C: torch.Tensor) -> torch.Tensor:
        """Calculate the voltage wave for a given scattering matrix.

        Parameters
        ----------
        S_C : torch.Tensor
            4d-tensor of shape [batch_size, nf, n0+n1, n0+n1]
            The scattering matrix of the connected circuitry

        Returns
        -------
        V_O_p : torch.Tensor
            voltage wave
            4d-tensor of shape [batch_size, nf, n0, n1]

        Notes
        -----
        This method calculates the incident complex voltage waves based on the given scattering matrix (in this case: the one of the circuitry).
        It uses z0, S_0, and nf attributes of the object to perform the calculation.

        The incident power is calculated as the absolute value squared of the voltage transfer function
        between the output port and the incident port. It is normalized by the characteristic impedance z0 and incident power.
        """

        S_0 = self.S_0
        n0 = self.S_0.shape[-1]
        z0 = self.z0
        P_inc = self.P_inc
        
        S_CL_11 = S_C[:, :, :n0, :n0]
        S_CL_12 = S_C[:, :, :n0, n0:]

        identity_matrix = torch.eye(n0).to(S_C.device)
        inverse_term = torch.inverse(identity_matrix - torch.matmul(S_CL_11, S_0))
        V_O_p = torch.matmul(inverse_term, S_CL_12) * torch.sqrt(torch.tensor(z0*P_inc))
        return V_O_p
    
    def calc_Pinc(self, V_O_p: torch.Tensor) -> torch.Tensor:
        """Calculate the incident power and phase for a given voltage wave.

        Parameters
        ----------
        V_O_p : torch.Tensor
            voltage wave
            4d-tensor of shape [batch_size, nf, n0, n1]

        Returns
        -------
        p_inc and phase : torch.Tensor
            4d-tensor of shape [batch_size, nf, n0, n1]
            incident power and phase.
        
        Notes
        -----
        The phase is calculated as the angle of the voltage transfer function between the output port
        and the incident port.
        """

        z0 = self.z0

        p_inc_nominator = torch.abs(V_O_p) ** 2
        p_inc_denominator = z0
        p_inc = p_inc_nominator / p_inc_denominator
        phase = torch.angle(V_O_p)
        return p_inc, phase

    @staticmethod
    def calc_norm(p_incM: torch.Tensor, 
                  phaseM: torch.Tensor) -> torch.Tensor:
        """Calculate the norm based on the incident power and phase.

        Parameters
        ----------
        p_incM : torch.Tensor
            4d-tensor of shape [batch_size, nf, n0, n1]
        phaseM : torch.Tensor
            4d-tensor of shape [batch_size, nf, n0, n1]

        Returns
        -------
        torch.Tensor
            4d-tensor of shape [batch_size, nf, n0, n1]
        """
        return torch.sqrt(p_incM)*torch.exp(1j*phaseM)
    
    @staticmethod
    def transform_fields_with_norm(field: torch.Tensor,
                                   norm: torch.Tensor) -> torch.Tensor:
        """Transform the fields with the norm.

        Parameters
        ----------
        field : torch.Tensor
            Takes either the electric or magnetic field with shape [nf, n0, 3, npoints]
        norm : torch.Tensor
            Takes the norm with shape [batch_size, nf, n0, n1]

        Returns
        -------
        torch.Tensor
            This results in a 4d-tensor of shape [batch_size, nf, n1, 3, npoints]  whereas 3 stands for the 3 cartesian components (x,y,z) of the field. The n1-fields can be combined further by the user (s. example with 8x1 Split-Ring Structure and Birdcage Coil).
        """
        # Preparation for broadcasting
        field = torch.moveaxis(field, 1, -1)    
        field = field.unsqueeze(0)
        field = field.unsqueeze(2)
        
        norm = norm.unsqueeze(-1)
        norm = torch.swapaxes(norm, 2, -2)
        norm = norm.unsqueeze(3)

        field = torch.matmul(field, norm)
        field = field.squeeze(-1)
        return field

    def calc_em_field(self, 
                      norm: torch.tensor) -> torch.Tensor: #TODO pass norm
            """Calculate the electromagnetic field.

            This method calculates the electromagnetic field based on the given incident power and phase.

            Parameters
            ----------
            norm : torch.Tensor
                4d-tensor of shape [batch_size, nf, n0, n1]

            Returns
            -------
            b_field, e_field : torch.Tensor
                4d-tensor of shape [batch_size, nf, n1, 3, npoints]
                The calculated electromagnetic fields.

            Notes
            -----
            As fields dependent linearly on the norm, the method will return the fields multiplied by the norm.
            If either the electric field or magnetic field is not given, the respective output will be None.
            """
            bfield = self.b_field
            efield = self.e_field

            if self.b_field is not None:
                bfield = self.transform_fields_with_norm(bfield, norm)

            if self.e_field is not None:
                efield = self.transform_fields_with_norm(efield, norm)

            return bfield, efield