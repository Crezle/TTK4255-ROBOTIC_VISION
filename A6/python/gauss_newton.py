import numpy as np

def compute_jacobian(resfun, p, epsilon):
    r = resfun(p)
    assert r.ndim == 1, '"resfun" must return a vector of scalars, but instead had shape %s' % str(r.shape)
    assert p.ndim == 1, '"p" must be a vector of scalars, but instead had shape %s' % str(p.shape)
    assert epsilon > 0.0, '"epsilon" must be non-zero and positive'
    J = np.empty((len(r), len(p)))
    for j in range(len(p)):
        pj0 = p[j]
        p[j] = pj0 + epsilon
        rpos = resfun(p)
        p[j] = pj0 - epsilon
        rneg = resfun(p)
        p[j] = pj0
        J[:,j] = rpos - rneg
    return J/(2.0*epsilon)

def gauss_newton(resfun, p0, step_size, num_steps, xtol=None, ftol=None, print_progress=False, finite_difference_epsilon=1e-6):
    jacfun = lambda p : compute_jacobian(resfun, p, epsilon=finite_difference_epsilon)
    r = resfun(p0)
    J = jacfun(p0)
    assert step_size > 0.0, '"step_size" must be non-zero and positive'
    assert num_steps > 0, '"num_steps" must be a positive integer'
    assert p0.ndim == 1, '"p0" must be a vector of scalars, but instead had shape %s' % str(p.shape)
    assert r.ndim == 1, '"resfun" must return a vector of scalars, but instead had shape %s' % str(r.shape)
    assert J.shape[0] == len(r), '"jacfun" must return a matrix with as many rows as there are elements in "resfun", but instead had shape %s' % str(J.shape)
    assert J.shape[1] == len(p0), '"jacfun" must return a matrix with as many columns as there are elements in "p0", but instead had shape %s' % str(J.shape)
    p = p0.copy()
    for iteration in range(num_steps):
        A = J.T@J
        b = -J.T@r
        d = np.linalg.solve(A, b)
        dp = step_size*d
        dE = (np.sum(resfun(p + dp)**2) - np.sum(resfun(p)**2))
        p += dp
        E = np.sum(resfun(p + dp)**2)
        if print_progress:
            if iteration == 0:
                print(' E          Change E   Change ||p||')
            print('%10.3e %10.3e %10.3e' % (E, np.linalg.norm(dE), np.linalg.norm(dp)))
        if xtol != None and np.linalg.norm(dp) <= xtol:
            print('Terminated after %d steps (xtol reached)' % iteration)
            break
        if ftol != None and np.linalg.norm(dE) <= ftol:
            print('Terminated after %d steps (ftol reached)' % iteration)
            break
        r = resfun(p)
        J = jacfun(p)
    return p
