import numpy as np
from tqdm import tqdm
import json, argparse, os
from scipy.ndimage import gaussian_filter

from src.field import Field
from src.system import System
from src.explicitTerms import Term
from src.fourierfunc import *

def main():

    initParser = argparse.ArgumentParser(description='model_Q_v_rho_alpha_newchi')
    initParser.add_argument('-s','--save_dir', help='directory to save data')
    initargs = initParser.parse_args()
    savedir = initargs.save_dir
    
    if os.path.isfile(savedir+"/parameters.json"):
	    with open(savedir+"/parameters.json") as jsonFile:
              parameters = json.load(jsonFile)
              
    run       = parameters["run"]
    T         = parameters["T"]        # final time
    dt_dump   = parameters["dt_dump"]
    n_steps   = int(parameters["n_steps"])  # number of time steps
    tauc      = parameters["tauc"]      # timescale of morphogen dissipation
    K         = parameters["K"]        # elastic constant, sets diffusion lengthscale of S with Gamma0
    alpha     = parameters["alpha"]    # active contractile stress
    beta      = parameters["beta"]
    chi       = parameters["chi"]      # coefficient of strain coupling  in Q's free energy
    gamma     = parameters["gamma"]    # coefficient of grad c coupling in Q's free energy
    lambd     = parameters["lambd"]   # lame
    mu        = parameters["mu"]       # lame
    B         = parameters["B"]        # elastic coupling constant
    D         = parameters["D"]        # diffusion constant for morphogen
    a         = parameters["a"]        # LdG constant
    b         = parameters["b"]        # LdG constant
    ct        = parameters["ct"]       # saturation constant for c coupling to Q
    epsilont  = parameters["epsilont"] # saturation constant for strain coupling to c
    epsilon   = 1e-3
    
    mx        = np.int32(parameters["mx"])
    my        = np.int32(parameters["my"])
    dx        = np.float32(parameters["dx"])
    dy        = np.float32(parameters["dy"])

    dt        = T / n_steps     # time step size
    n_dump    = round(T / dt_dump)
    dn_dump   = round(n_steps / n_dump)
    
     # Define the grid size.
    grid_size = np.array([mx, my])
    dr=np.array([dx, dy])

    k_list, k_grids = momentum_grids(grid_size, dr)
    fourier_operators = k_power_array(k_grids)
    # Initialize the system.
    system = System(grid_size, fourier_operators)

    # Create fields that undergo timestepping
    system.create_field('c', k_list, k_grids, dynamic=True)
    system.create_field('Qxx', k_list, k_grids, dynamic=True)
    system.create_field('Qxy', k_list, k_grids, dynamic=True)
    
    # Create fields that don't timestep
    system.create_field('Ident', k_list, k_grids, dynamic=False)
    system.create_field('S2', k_list, k_grids, dynamic=False)

    #system.create_field('epsilon_xy', k_list, k_grids, dynamic=False)
    #system.create_field('epsilon_xx', k_list, k_grids, dynamic=False)
    #system.create_field('epsilon_yy', k_list, k_grids, dynamic=False)
    system.create_field('epsilon_iso', k_list, k_grids, dynamic=False)
    system.create_field('epsilon_e', k_list, k_grids, dynamic=False)

    system.create_field('iqxc', k_list, k_grids, dynamic=False)
    system.create_field('iqyc', k_list, k_grids, dynamic=False)
    system.create_field('satc', k_list, k_grids, dynamic=False)

    system.create_field('Hxx', k_list, k_grids, dynamic=False)
    system.create_field('Hxy', k_list, k_grids, dynamic=False)
    
    # Create equations, if no function of rho, write None. If function and argument, supply as a tuple. 
    # Write your own functions in the function library or use numpy functions
    # if using functions that need no args, like np.tanh, write [("fieldname", (np.tanh, None))]
    # for functions with an arg, like, np.power(fieldname, n), write [("fieldname", (np.power, n))]
    # Define Identity # The way its written if you don't define a RHS, the LHS becomes zero at next timestep for Static Fields
    system.create_term("Ident", [("Ident", None)], [1, 0, 0, 0, 0])
    # Define S2
    system.create_term("S2", [("Qxx", (np.square, None))], [4.0, 0, 0, 0, 0])
    system.create_term("S2", [("Qxy", (np.square, None))], [4.0, 0, 0, 0, 0])
    # Define epsilon_iso
    system.create_term("epsilon_iso", [("Qxx", None)], [alpha/(2*mu + lambd), 0, 2, 0, 1])
    system.create_term("epsilon_iso", [("Qxx", None)], [-alpha/(2*mu + lambd), 0, 0, 2, 1])
    system.create_term("epsilon_iso", [("Qxy", None)], [alpha/(2*mu + lambd), 0, 1, 1, 1])
    
    system.create_term("epsilon_e", [("epsilon_iso", None)], [1/epsilont, 0, 0, 0, 0])
    system.create_term("epsilon_e", [("Ident", None)], [-1, 0, 0, 0, 0])
    # Define gradients of c
    system.create_term("iqxc", [("c", None)], [1, 0, 1, 0, 0])
    system.create_term("iqyc", [("c", None)], [1, 0, 0, 1, 0])
    system.create_term("satc", [("Ident", None)], [ct*ct, 0, 0, 0, 0])
    system.create_term("satc", [("iqxc", (np.power,2))], [1, 0, 0, 0, 0])
    system.create_term("satc", [("iqyc", (np.power,2))], [1, 0, 0, 0, 0])
    # Define Hxx
    system.create_term("Hxx", [("Qxx", None)], [a, 0, 0, 0, 0])
    system.create_term("Hxx", [("S2", None), ("Qxx", None)], [-b, 0, 0, 0, 0])
    system.create_term("Hxx", [("Qxx", None)], [-K, 1, 0, 0, 0])
    system.create_term("Hxx", [("iqxc", (np.power, 2)), ("satc", (np.power, -1))], [gamma/2, 0, 0, 0, 0])
    system.create_term("Hxx", [("iqyc", (np.power, 2)), ("satc", (np.power, -1))], [-gamma/2, 0, 0, 0, 0])
    #system.create_term("Hxx", [("epsilon_xx", None)], [chi/2, 0, 0, 0, 0])
    #system.create_term("Hxx", [("epsilon_yy", None)], [-chi/2, 0, 0, 0, 0])
    # Define Hxy
    system.create_term("Hxy", [("Qxy", None)], [a, 0, 0, 0, 0])
    system.create_term("Hxy", [("S2", None), ("Qxy", None)], [-b, 0, 0, 0, 0])
    system.create_term("Hxy", [("Qxy", None)], [-K, 1, 0, 0, 0])
    system.create_term("Hxy", [("iqxc", None), ("iqyc", None), ("satc", (np.power, -1))], [gamma, 0, 0, 0, 0])
    #system.create_term("Hxy", [("epsilon_xy", None)], [chi, 0, 0, 0, 0])

    # Create terms for rho timestepping
    system.create_term("c", [("c", None)], [-1/tauc, 0, 0, 0, 0])
    system.create_term("c", [("c", None)], [-D, 1, 0, 0, 0])
    system.create_term("c", [("epsilon_e", (np.heaviside, 0))], [beta, 0, 0, 0, 0])

    # Create terms for Qxx timestepping
    system.create_term("Qxx", [("Hxx", None)], [1, 0, 0, 0, 0])

    # Create terms for Qxy timestepping
    system.create_term("Qxy", [("Hxy", None)], [1, 0, 0, 0, 0])
    
    c       = system.get_field('c')
    Qxx     = system.get_field('Qxx')
    Qxy     = system.get_field('Qxy')
    #exx      = system.get_field('epsilon_xx')
    #eyy      = system.get_field('epsilon_yy')
    eiso      = system.get_field('epsilon_iso')

    # Initialise c
    c.set_real(0.01*np.random.rand(mx, my)+0.02)
    c.synchronize_momentum()
    
    set_aster(Qxx, Qxy, grid_size, dr, nem_length=np.sqrt(K/a))

    # Initialise identity matrix 
    system.get_field('Ident').set_real(np.ones(shape=grid_size))
    system.get_field('Ident').synchronize_momentum()
    # Initialise identity matrix 
    system.get_field('satc').set_real(np.ones(shape=grid_size))
    system.get_field('satc').synchronize_momentum()

    if not os.path.exists(savedir+'/data/'):
        os.makedirs(savedir+'/data/')

    for t in tqdm(range(n_steps)):

        if t % dn_dump == 0:
            np.savetxt(savedir+'/data/'+'c.csv.'+ str(t//dn_dump), c.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'Qxx.csv.'+ str(t//dn_dump), Qxx.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'Qxy.csv.'+ str(t//dn_dump), Qxy.get_real(), delimiter=',')
            np.savetxt(savedir+'/data/'+'eiso.csv.'+ str(t//dn_dump), eiso.get_real(), delimiter=',')
            #np.savetxt(savedir+'/data/'+'eyy.csv.'+ str(t//dn_dump), eyy.get_real(), delimiter=',')
            #np.savetxt(savedir+'/data/'+'exy.csv.'+ str(t//dn_dump), exy.get_real(), delimiter=',')

        system.update_system(dt)

def momentum_grids(grid_size, dr):
    k_list = [np.fft.fftfreq(grid_size[i], d=dr[i])*2*np.pi for i in range(len(grid_size))]
    # k is now a list of arrays, each corresponding to k values along one dimension.

    k_grids = np.meshgrid(*k_list, indexing='ij')
    #k_grids = np.meshgrid(*k_list, indexing='ij', sparse=True)
    # k_grids is now a list of 2D sparse arrays, each corresponding to k values in one dimension.

    return k_list, k_grids

def k_power_array(k_grids):
    k_squared = sum(ki**2 for ki in k_grids)
    #k_squared = k_grids[0]**2 + k_grids[1]**2
    k_abs = np.sqrt(k_squared)
    inv_kAbs = np.divide(1.0, k_abs, where=k_abs!=0)

    k_power_arrays = [k_squared, 1j*k_grids[0], 1j*k_grids[1], inv_kAbs]

    return k_power_arrays

def pade(r):
    return r*np.sqrt((0.34+(0.07*r**2))/(1+(0.41*r**2)+(0.07*r**4)))

def set_aster(fieldxx, fieldxy, size, dr, nem_length):
    center1 = dr*size/2
    center2 = center1 - np.array([0, nem_length*15])
    center3 = center1 + np.array([0, nem_length*15])

    tol = 0.001
    x   = np.arange(0+tol, dr[0]*size[0]-tol, dr[0])
    y   = np.arange(0+tol, dr[1]*size[1]-tol, dr[1])
    r   = np.meshgrid(x,y, indexing='ij')

    S1 = pade(((r[0]-center1[0])**2 + (r[1]-center1[1])**2)/nem_length)
    S2 = pade(((r[0]-center2[0])**2 + (r[1]-center2[1])**2)/nem_length)
    S3 = pade(((r[0]-center3[0])**2 + (r[1]-center3[1])**2)/nem_length)
    
    theta1 = np.arctan2((r[1]-center1[1]),(r[0]-center1[0]))
    theta2 = -np.arctan2((r[1]-center2[1]),(r[0]-center2[0]))/2
    theta3 = -np.arctan2((r[1]-center3[1]),(r[0]-center3[0]))/2

    theta = theta1+theta2+theta3
    S = S1*S2*S3

    Qxx = S*np.cos(theta*2)/2
    Qxy = S*np.sin(theta*2)/2
    
    fieldxx.set_real(Qxx)
    fieldxx.synchronize_momentum()
    fieldxy.set_real(Qxy)
    fieldxy.synchronize_momentum()

if __name__=="__main__":
    main()
