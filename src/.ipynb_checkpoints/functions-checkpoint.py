from numpy import power

def hilln(x, n, x0):
    """
    Given an array x and a power n, return a Hill function of nth power.
    """
    return power(x, n)/(power(x0, n)+power(x, n))

def logistic(x, x_end):
    """
    Given an array x, and a float ending density x_end, return logistic function.
    """
    return x*(1-(x/x_end))