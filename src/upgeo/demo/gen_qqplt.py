
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib import rc
from upgeo.demo.util import loadmat_data


def plot_qq(pred, actual, fname=None):
    rc('text', usetex=True)
    rc('font', family='serif')
    
    plt.figure()
    plt.plot(actual, pred, 'wp')
    xmin,xmax = plt.xlim()
    ymin,ymax = plt.ylim()
    
    #print 'xmin={0},xmax={1}'.format(xmin,xmax)
    #print 'ymin={0},ymax={1}'.format(ymin,ymax)
    
    min = np.min([xmin,ymin])
    max = np.max([xmax,ymax])
    plt.xlim(min,max)
    plt.ylim(min,max)
    plt.plot(np.arange(min,max+1),np.arange(min,max+1), color='k', linestyle='-', linewidth=2)
    plt.ylabel(r'predicted', fontsize=24)
    plt.xlabel(r'actual', fontsize=24)
    plt.grid(True, 'major')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.xticks(np.arange(xmin,xmax+1,0.5))
    if fname == None:
        plt.show()
    else:
        plt.savefig(fname + ".eps")
        plt.savefig(fname + ".png")
        
def ensure_dir(pathname):
    if not os.path.exists(pathname):
        os.makedirs(pathname)

if __name__ == '__main__':
    
    datafile = '/home/marcel/datasets/multilevel/eusinan/bssa/eval_eudata.mat'
    evalpath = '/home/marcel/datasets/multilevel/eusinan/bssa/results/T4/splitz1'  
    
    
    X,Y = loadmat_data(datafile)
    
    actual = Y[:,0]
    mean = np.mean(actual)
    
    ensure_dir(os.path.join(evalpath, 'plots'))
    for file in os.listdir(evalpath):
        if file != 'plots':
            print 'file: {0}'.format(file)
            result = np.loadtxt(os.path.join(evalpath, file), delimiter=',')
            if not 'testerror' in file:
                pred = result[:,0] + mean
            plot_qq(pred, actual, fname=os.path.join(evalpath,'plots',file+'_qq'))