import pandas as pd 
import numpy as np
import sympy as sp


def total_length_of_curve(start, final, manifold):
    '''
    This function compute the length of the curve. 
    WE recognize the problem of using linspace when
    we often deal with logaritmhic scale, but we wasn't
    able to find other way
    '''
    epsilon = 1e-6
    x_previous = np.arange(start,final,epsilon)
    y_previous = manifold(x_previous)
    x = np.delete(np.append(x_previous, [final]), 0)
    y = np.delete(np.append( y_previous, manifold(final)), 0)
    #breakpoint()
    distances = np.sqrt((x-x_previous)**2+(y-y_previous)**2)
    #distances = np.sqrt(1+((y-y_previous)/(x-x_previous))**2)
    #distances = np.abs(y-y_previous)
    s =[]
    for i, value in enumerate(distances):
    #length = np.sum(distances)
        s.append(np.sum(distances[:i]))
    #return length
    return np.array(s)
def function_spaced(start, final, n_point, manifold):
    '''
    Get equispaced points on a curve.
    It find the length of the curve as sum of 
    small segment.
    '''
    epsilon = 1e-2

    length = total_length_of_curve(start,final,manifold)
    length_step = length/n_point
    #breakpoint()
    well_spaced_points = []
    x = np.geomspace(start,final,10)
    print("the computed length of the curve is", length )
    print("the step is ",length_step)
    for point in x:
        if np.isclose(total_length_of_curve(start, point, manifold), length_step, atol = 1e-2):
            start = point
            #breakpoint()
            print("new point: ", point)
            well_spaced_points.append(point)
    return np.array(well_spaced_points )   

if __name__ == "__main__":
    import matplotlib.pylab as plt
    manifold = lambda x:np.arctan(x)
    x = function_spaced(1,100, 40, manifold)
    plt.plot(x, manifold(x), "o")
    plt.show()
