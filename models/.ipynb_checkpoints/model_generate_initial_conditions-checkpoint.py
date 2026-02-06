import numpy as np
from tqdm import tqdm
import json, argparse, os

from src.field import Field
from src.system import System
from src.explicitTerms import Term
from src.fourierfunc import *

def main():

    initParser = argparse.ArgumentParser(description='init conditions')
    initParser.add_argument('-s','--save_dir', help='directory to save data')
    initargs = initParser.parse_args()
    savedir = initargs.save_dir
    
    if os.path.isfile(savedir+"/parameters.json"):
	    with open(savedir+"/parameters.json") as jsonFile:
              parameters = json.load(jsonFile)
              
    run       = parameters["run"]
    T         = 4
    n_steps   = 4000
    a         = parameters["a"]
    d         = parameters["d"]
    rho_in    = parameters["rho_in"]   # isotropic to nematic transition density, or "onset of order in the paper"
    rho_min   = parameters["rho_min"] /rho_in #minimum density to be reached while phase separating
    rho_max   = parameters["rho_max"] /rho_in #minimum density to be reached while phase separating
    rho_seed  = parameters["rhoseed"] /rho_in     # seeding density, normalised by 100 mm^-2
    mx        = np.int32(parameters["mx"])
    my        = np.int32(parameters["my"])
    dx        = np.float32(parameters["dx"])
    dy        = np.float32(parameters["dy"])

    dt        = T / n_steps     # time step size
    
     # Define the grid size.
    grid_size = np.array([mx, my])
    dr=np.array([dx, dy])

    k_list, k_grids = momentum_grids(grid_size, dr)
    fourier_operators = k_power_array(k_grids)
    # Initialize the system.
    system = System(grid_size, fourier_operators)

    # Create fields that undergo timestepping
    system.create_field('rho', k_list, k_grids, dynamic=True)

    system.create_field('Ident', k_list, k_grids, dynamic=False)
    system.create_field('mu', k_list, k_grids, dynamic=False)
    
    # Create equations, if no function of rho, write None. If function and argument, supply as a tuple. 
    # Write your own functions in the function library or use numpy functions
    # if using functions that need no args, like np.tanh, write [("fieldname", (np.tanh, None))]
    # for functions with an arg, like, np.power(fieldname, n), write [("fieldname", (np.power, n))]
    # Define Identity # The way its written if you don't define a RHS, the LHS becomes zero at next timestep for Static Fields
    system.create_term("Ident", [("Ident", None)], [1, 0, 0, 0])
    # Define chemical potential for rho, it phase separates into 0 and rho_c
        #first cahn-hilliard terms
    system.create_term("mu", [("rho", (np.power, 3))], [4, 0, 0, 0])
    system.create_term("mu", [("rho", (np.power, 2))], [-6*(rho_min+rho_max), 0, 0, 0])
    system.create_term("mu", [("rho", None)], [2*(rho_max**2 + rho_min**2 + 4*rho_min*rho_max), 0, 0, 0])
    system.create_term("mu", [("Ident", None)], [-2*(rho_min + rho_max)*rho_min*rho_max, 0, 0, 0])
    system.create_term("mu", [("rho", None)], [d, 1, 0, 0])

    # Create terms for rho timestepping
        # phase separation, separates into 0 and rho_c
    system.create_term("rho", [("mu", None)], [-a, 1, 0, 0])
    
    rho     = system.get_field('rho')
    # set init condition and synchronize momentum with the init condition, important!!
    #set_rho_islands(rho, int(100*rho_seed/0.16), rho_seed, grid_size)
    rhoseed = np.random.rand(mx, my) + np.ones([mx, my])
    rhoseed = rhoseed*rho_seed/np.average(rhoseed)
    rho.set_real(rhoseed)
    rho.synchronize_momentum()

    # Initialise identity matrix 
    system.get_field('Ident').set_real(np.ones(shape=grid_size))
    system.get_field('Ident').synchronize_momentum()

    if not os.path.exists(savedir+'/data/'):
        os.makedirs(savedir+'/data/')

    for t in tqdm(range(n_steps)):
        system.update_system(dt)
    #save the last frame as initial condition for main simulation
    rho_save = rho.get_real()
    if np.sum(np.where(rho_save<0, 1, 0)) > 0 :
        rho_save -= np.min(rho_save)
    np.savetxt(savedir+'/data/'+'rho.csv.'+ str(-1), rho_save, delimiter=',')
            
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
    #inv_kAbs = np.divide(1.0, k_abs, where=k_abs!=0)

    k_power_arrays = [k_squared, 1j*k_grids[0], 1j*k_grids[1]]

    return k_power_arrays

if __name__=="__main__":
    main()
