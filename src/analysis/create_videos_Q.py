import numpy as np
import json, argparse, os
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox


def main():

    initParser = argparse.ArgumentParser(description='model_Q_v_rho_create_videos')
    initParser.add_argument('-s','--save_dir', help='directory to save data')
    initargs = initParser.parse_args()
    savedir = initargs.save_dir

    if not os.path.exists(savedir+'/videos/'):
        os.makedirs(savedir+'/videos/')
    
    if os.path.isfile(savedir+"/parameters.json"):
        with open(savedir+"/parameters.json") as jsonFile:
              parameters = json.load(jsonFile)
    
    T         = parameters["T"]        # final time
    dt_dump   = parameters["dt_dump"]
    n_steps   = int(parameters["n_steps"])  # number of time steps
    dt        = T / n_steps     # time step size
    n_dump    = round(T / dt_dump)    
    mx        = np.int32(parameters["mx"])
    my        = np.int32(parameters["my"])
    dx        = np.float32(parameters["dx"])
    dy        = np.float32(parameters["dy"])
    Lx        = mx*dx
    Ly        = my*dy
    
    #setup a meshgrid
    tol = 0.001
    
    x   = np.linspace(0+tol, Lx-tol, mx)
    y   = np.linspace(0+tol, Ly-tol, my)
    xv, yv  = np.meshgrid(x,y, indexing='ij')
    
    times = np.arange(0, n_dump, 1)*dt_dump

    
    figc, axc= plt.subplots(figsize=(12, 8), ncols=1)
    figS, axS= plt.subplots(figsize=(12, 8), ncols=1)

    n=1
    p_factor = np.int32(mx/39)
    
    c   = np.loadtxt(savedir+'/data/'+'c.csv.{:d}'.format(n), delimiter=',')
    Qxx = np.loadtxt(savedir+'/data/'+'Qxx.csv.{:d}'.format(n), delimiter=',')
    Qxy = np.loadtxt(savedir+'/data/'+'Qxy.csv.{:d}'.format(n), delimiter=',')
    pxv = xv[p_factor:-1:p_factor, p_factor:-1:p_factor]
    pyv = yv[p_factor:-1:p_factor, p_factor:-1:p_factor]
    S   = 2*np.sqrt(Qxx**2+Qxy**2)
    theta = np.arctan2(Qxy, Qxx)/2
    nx    = np.cos(theta) [p_factor:-1:p_factor, p_factor:-1:p_factor]
    ny    = np.sin(theta) [p_factor:-1:p_factor, p_factor:-1:p_factor]
    Snx   = S [p_factor:-1:p_factor, p_factor:-1:p_factor] * nx
    Sny   = S [p_factor:-1:p_factor, p_factor:-1:p_factor] * ny
    vscale = 0.05
    nscale = 0.3
    
    cc = [axc.pcolormesh(xv, yv, c, cmap='viridis', vmin=0), axc.quiver(pxv, pyv, Snx, Sny, color='b', pivot='middle', headlength=0, headaxislength=0, scale=nscale, scale_units='xy')]
    cS   = [axS.pcolormesh(xv, yv, S, cmap='viridis', vmin=0, vmax=2.0), axS.quiver(pxv, pyv, nx, ny, color='k', pivot='middle', headlength=0, headaxislength=0)]

    figc.colorbar(cc[0])
    axc.set_title('Morphogen')
    figS.colorbar(cS[0])
    axS.set_title('Nematic Order')
    
    tbaxc = figc.add_axes([0.2, 0.93, 0.04, 0.04])
    tbc   = TextBox(tbaxc, 'time')
    tbaxS = figS.add_axes([0.2, 0.93, 0.04, 0.04])
    tbS   = TextBox(tbaxS, 'time')    
    
    def plt_snapshot_c(val):        
        c   = np.loadtxt(savedir+'/data/'+'c.csv.{:d}'.format(val), delimiter=',')
        Qxx = np.loadtxt(savedir+'/data/'+'Qxx.csv.{:d}'.format(val), delimiter=',')
        Qxy = np.loadtxt(savedir+'/data/'+'Qxy.csv.{:d}'.format(val), delimiter=',')
        S   = 2*np.sqrt(Qxx**2+Qxy**2) 
        theta = np.arctan2(Qxy, Qxx)/2
        Snx    = (S*np.cos(theta)) [p_factor:-1:p_factor, p_factor:-1:p_factor]
        Sny    = (S*np.sin(theta)) [p_factor:-1:p_factor, p_factor:-1:p_factor]
        
        cc[0].set_array(c)
        cc[1].set_UVC(Snx, Sny)
        tbc.set_val(round(times[val],2))
        
        figc.canvas.draw_idle()
    
    def plt_snapshot_S(val):
        Qxx = np.loadtxt(savedir+'/data/'+'Qxx.csv.{:d}'.format(val), delimiter=',')
        Qxy = np.loadtxt(savedir+'/data/'+'Qxy.csv.{:d}'.format(val), delimiter=',')
        S   = 2*np.sqrt(Qxx**2+Qxy**2)
        theta = np.arctan2(Qxy, Qxx)/2
        nx    = np.cos(theta)
        ny    = np.sin(theta)
        
        cS[0].set_array(S)
        cS[1].set_UVC(nx[p_factor:-1:p_factor, p_factor:-1:p_factor], ny[p_factor:-1:p_factor, p_factor:-1:p_factor])
        tbS.set_val(round(times[val],2))

        figS.canvas.draw_idle()

    
    from matplotlib.animation import FuncAnimation
    animc = FuncAnimation(figc, plt_snapshot_c, frames = n_dump, interval=100, repeat=True)
    animc.save(savedir+'/videos/'+'c.mp4')

    anime = FuncAnimation(figS, plt_snapshot_S, frames = n_dump, interval=100, repeat=True)
    anime.save(savedir+'/videos/'+'Q.mp4')

if __name__=="__main__":
    main()
