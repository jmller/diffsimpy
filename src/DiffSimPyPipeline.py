from DiffSimPy import DiffCoSimulator

class CalculationPipeline(DiffCoSimulator):
    """
    A pipeline for performing simulation calculations using DiffCoSimulator. 
    
    This class extends the DiffCoSimulator and performs forward calculations 
    based on passed S-Matrices and magnetic or electric fields.

    Attributes:
    -----------
    S_0 : torch.Tensor 
        2d-tensor of shape [n0, n0]; S-Matrix of simulated resonant structure from em-simulation (unconnected structure); n_0 is the number of ports.
    S_C_const : torch.Tensor
        4d-tensor of shape [batch_size, nf, n0+n1, n0+n1]; S-Matrix of the circuitry to be connected to the resonant structure, Explanation follows in Notes. Follows standards defined by CoSimPy.
    b_field or e_field : Union[torch.Tensor, None], optional
        4d-tensor of shape [nf, n0, 3, npoints]; Magnetic or electric fields of individual ports, by default None.
    **kwarg : additional arguments
        Additional keyword arguments passed to the DiffCoSimulator.

    Methods:
    --------
    __init__(self, S_0, S_C_const, b_field=None, e_field=None, **kwarg):
        Initializes the CalculationPipeline with the given parameters, 
        including optional magnetic and electric fields.
        
    forward(self, s_params, indices: tuple = None):
        Performs a forward pass through the pipeline, calculating 
        the electromagnetic fields using the simulation parameters.
    """
    
    def __init__(self, S_0, S_C_const, b_field=None, e_field=None, **kwarg): #TODO: initializes z0=50;allow for other values
        """ Initialize the CalculationPipeline instance.
        """
        super(CalculationPipeline, self).__init__(S_0=S_0, 
                                                   S_C_const=S_C_const,
                                                   b_field=b_field, 
                                                   e_field=e_field, **kwarg)

    def forward(self, s_params, indices: tuple = None):
        """ Perform a forward calculation to calculate the electromagnetic fields.
        
        Parameters:
        -----------
        s_params : torch.tensor
            flattened tensor of scattering parameters that wlil be assigned to the S_C matrix by the indices sc_idx and sc_idy
        indices : tuple, optional
            A tuple containing indices of the scattering parameters that are required for assignement of individual s_params elements to the S_C matrix. The length of sc_idx and sc_idy has to be equal to the number of scattering parameters in s_params.
        
        Returns:
        --------
        b_field : magnetic field
            The computed magnetic field based on the given parameters.
        e_field : electric field
            The computed electric field based on the given parameters.

        Workflow:
        ---------
        1. Assign s-params to S_C_var `s_params` and combine with S_C_const if `indices` are provided.
        2. Build S_C.
        3. Calculate the voltage wave.
        4. Compute the incident power and phase.
        5. Calculate the norm for further processing.
        6. Calculate the electromagnetic fields.
        """
        if indices != None:
            S_C_var = self.assign_S_C_var(s_params=s_params, 
                                          sc_idx=indices[0], 
                                          sc_idy=indices[1])
            S_C = self.build_S_C(S_C_var)
        else:
            S_C = self.S_C_const
        V_O_p = self.calc_voltage_wave(S_C)
        p_incM, phaseM = self.calc_Pinc(V_O_p)
        norm = self.calc_norm(p_incM, phaseM)
        b_field, e_field = self.calc_em_field(norm)
        return b_field, e_field
