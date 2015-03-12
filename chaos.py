import numpy as np
import matplotlib.pyplot as plt
pi = np.pi


#Quick method for doing divided difference on data to 
#establish a rate a change, given a desired step size to look at change over
def divdiff(step, data):
    diff = data[:len(data)-step, :] - data[step:len(data), :]
    return diff[:, 0] / diff[:, 1]

#Given a dataset, you can reinflate state space to
#understand topology of space
#m=dimensions, t=tau (step size), vert&horz are axis projections
def reconstruct_space(data, m, t, vert, horz):
    recon = []
    for i in xrange(len(data)-(t*m)):
        comb = []
        for j in xrange(m):
            comb.append(data[i+(t*j)])
        recon.append(np.array(comb))
    theta = []
    theta_m = []
    for state in recon:
        theta.append(state[vert,0])
        theta_m.append(state[horz,0])
    return theta, theta_m


#4th-order Runge-Kutta
#f = array of functions for derivatives, x = initial conditions for state vector
#t = starting time, h = time step, and n = number of iterations
#Use np arrays to guarantee proper matrix operations for vectors

def rk4(f, x, t, h, n, save_me=False, path=None):
    dims = len(x[0])
    k1 = np.zeros([1, dims])
    k2 = np.zeros([1, dims])
    k3 = np.zeros([1, dims])
    k4 = np.zeros([1, dims])
    x_final = np.zeros([dims,n])
    it = 0
    while it < n:
        for i in range(dims):
            k1[0, i] = (f[i](x, t))
        for i in range(dims):
            k2[0, i] = (f[i](x + .5*h*k1, t + h/2.))
        for i in range(dims):
            k3[0, i] = (f[i](x + .5*h*k2, t + h/2.))
        for i in range(dims):
            k4[0, i] = (f[i](x + h*k3, t + h))
        x_next = x + ((h/6.) * (k1 + 2.*k2 + 2.*k3 + k4))
        for i in range(dims):
            x_final[i , it] = x_next[0, i]
        x = x_next
        t = t+h
        it += 1

    if save_me:
        with open(path, 'w') as fs:
            fs.write(x_final)
            fs.close()

    return x_final

#Force Driven Damped Pendulum; returns a 2D array of functions
#a = alpha (frequency), b = beta (drag), m = mass, l = length, A = drive amplitude
#The functions will anticipate an np.array with an angle [0] and angular velocity [1]
def damped_pen(a, b, m, l, A):
    theta_dot = (lambda x, t: x[0, 1])
    omega_dot = (lambda x, t: ((A*np.cos(a*t)) - (b*l*x[0, 1]) - (m*9.8*np.sin(x[0, 0]))) / (m*l))
    return np.array([theta_dot, omega_dot])

def simple():
    y = lambda x, t: 2*x
    return np.array([y])

#Provide a set of initial conditions to try for making a portrait
#Also can change parameters of the state equation
#L = list of np.arrays that have initial conditions
def create_portrait(a, b, m, l, A, mod=False, adapt=False):
    my_eqs = damped_pen(a, b, m, l, A)
    portrait_points = []
    L = []
    for i in range(0,50, 10):
        L.append(np.array([[-3*pi, i]]))
        L.append(np.array([[3*pi, -i]]))
        # L.append(np.array([[-pi, i]]))
        # L.append(np.array([[pi, -i]]))
        # L.append(np.array([[-pi, -i]]))
        # L.append(np.array([[2, i]]))
        # L.append(np.array([[-2, i]]))
        # L.append(np.array([[-1.5, -i]]))
        # L.append(np.array([[-.5, -i]]))
        # L.append(np.array([[3.0*pi/2., i]]))
        # L.append(np.array([[-3.0*pi/2., i]]))
        # L.append(np.array([[3.0*pi/2., -i]]))
        # L.append(np.array([[-3.0*pi/2., -i]]))
        # L.append(np.array([[-3.0*pi, i]]))
        # L.append(np.array([[-2.0*pi, i]]))
        # L.append(np.array([[-5.0*pi/2., i]]))
        # L.append(np.array([[-5.0*pi/2., -i]]))
    plt.figure(figsize=(15, 10), dpi=80)
    plt.axis([0, 2*pi, -50, 50])
    if adapt:
        for ic in L:
            state_data = rk4_adapt_solver(my_eqs, ic, 0, .005, 2000)
            portrait_points.append(state_data)
    else:
        for ic in L:
            state_data = rk4(my_eqs, ic, 0, .005, 2000)
            portrait_points.append(state_data)
    if not mod:
        for possibles in portrait_points:
            plt.plot(possibles[0, :], possibles[1, :], 'b-')
    else:
        for possibles in portrait_points:
            mod_theta = map((lambda x: x%(2*pi)), possibles[0, :])
            plt.plot(mod_theta, possibles[1, :], 'b-')

#Single RK4 step
def rk4_step(f, x, t, h):
    dims = len(x[0])
    k1 = np.zeros([1, dims])
    k2 = np.zeros([1, dims])
    k3 = np.zeros([1, dims])
    k4 = np.zeros([1, dims])

    for i in range(dims):
        k1[0, i] = (f[i](x, t))
    for i in range(dims):
        k2[0, i] = (f[i](x + .5*h*k1, t + h/2.))
    for i in range(dims):
        k3[0, i] = (f[i](x + .5*h*k2, t + h/2.))
    for i in range(dims):
        k4[0, i] = (f[i](x + h*k3, t + h))
    x_next = x + ((h/6.) * (k1 + 2.*k2 + 2.*k3 + k4))

    return x_next

#Adaptive timestep RK4 solver
def rk4_adapt_solver(f, x, t, h0, n, tol=.01):
    dims = len(x[0])
    x_final = np.zeros([dims,n])
    it = 0
    while it < n:
        h_test1 = rk4_step(f, x, t, h0)
        h_test12 = rk4_step(f, h_test1, t+h0, h0)
        h_test2 = rk4_step(f, x, t, 2.*h0)
        diff = np.linalg.norm( (h_test2 - h_test12) )
        if diff > tol:
            h0 = h0 * ((tol/diff)**(1./5.))
            x_next = rk4_step(f, x, t, h0)
        else:
            x_next = h_test2
        for i in range(dims):
            x_final[i , it] = x_next[0, i]
        x = x_next
        t = t+h0
        it += 1

    return x_final

#Equations for Lorenz attractor
def lorenz(a, r, b):
    x_dot = (lambda x, t: a * (x[0,1] - x[0,0]))
    y_dot = (lambda x, t: (r * x[0,0]) - x[0,1] - (x[0,0] * x[0,2]))
    z_dot = (lambda x, t: (x[0,0] * x[0,1]) - (b * x[0,2]))
    return [x_dot, y_dot, z_dot]

#Variational equations for Lorenz
def lorenz_var(a, r, b):
    x_dot = (lambda x, t: a * (x[0,1] - x[0,0]))
    y_dot = (lambda x, t: (r * x[0,0]) - x[0,1] - (x[0,0] * x[0,2]) )
    z_dot = (lambda x, t: (x[0,0] * x[0,1]) - (b * x[0,2]))
    dxx = (lambda x, t: -a*x[0,3] + a*x[0,4] )
    dxy = (lambda x, t: (r - x[0,2])*x[0,3] - x[0,4] - x[0,0]*x[0,5] )
    dxz = (lambda x, t: x[0,1]*x[0,3] + x[0,0]*x[0,4] - b*x[0,5] )
    dyx = (lambda x, t: -a*x[0,6] + a*x[0,7] )
    dyy = (lambda x, t: (r - x[0,2])*x[0,6] - x[0,7] - x[0,0]*x[0,8] )
    dyz = (lambda x, t: x[0,1]*x[0,6] + x[0,0]*x[0,7] - b*x[0,8] )
    dzx = (lambda x, t: -a*x[0,9] + a*x[0,10] )
    dzy = (lambda x, t: (r - x[0,2])*x[0,9] - x[0,10] - x[0,0]*x[0,11] )
    dzz = (lambda x, t: x[0,1]*x[0,9] + x[0,0]*x[0,10] - b*x[0,11] )
    return [x_dot, y_dot, z_dot, dxx, dxy, dxz, dyx, dyy, dyz, dzx, dzy, dzz]

#Rossler attractor equations
def rossler(a, b ,c):
    x_dot = (lambda x, t: -(x[0,1] + x[0,2]))
    y_dot = (lambda x, t: x[0,0] + (a*x[0,1]))
    z_dot = (lambda x, t: b + (x[0,2] * (x[0,0] - c)))
    return [x_dot, y_dot, z_dot]

#Take a temporal poincare section of a state-space trajectory
#traj=trajectory data, T=hyperplane
def temp_poincare_simple(traj, T):
    #Each dimension of the state trajectory should be a separate row
    #time should have its own row in the matrix
    rows = len(traj)
    dims = rows - 1

    poincare_section = []

    for i in range(len(traj[0]) - 1):
        x1 = traj[:,i]
        x2 = traj[:,i+1]

        t1 = x1[rows - 1]
        t2 = x2[rows - 1]

        test1 = t1 % (2*T)
        test2 = t2 % (2*T)

        if (test1 < T and test2 > T) or (test1 > T and test2 < T):
            poincare_section.append(x2)

    return poincare_section

#Smarter poincare section that interpolates hyperplane piercing
def temp_poincare_smart(traj, T):
    #Each dimension of the state trajectory should be a separate row
    #time should have its own row in the matrix
    rows = len(traj)
    dims = rows - 1

    poincare_section = []

    for i in range(len(traj[0]) - 1):
        x1 = traj[:,i]
        x2 = traj[:,i+1]

        t1 = x1[rows - 1]
        t2 = x2[rows - 1]

        test1 = t1 % T
        test2 = t2 % T

        if test1 > test2:
            x_diff = x2 - x1
            lin_adj = (T - test1) / ((T+test2) - test1)
            x_inter = (lin_adj*x_diff) + x1
            poincare_section.append(x_inter)

    return poincare_section

#RK4 solver that also keeps track of time rather than implicit
#time tracking
def rk4_wtime(f, x, t, h, n, save_me=False, path=None):
    dims = len(x[0])
    k1 = np.zeros([1, dims])
    k2 = np.zeros([1, dims])
    k3 = np.zeros([1, dims])
    k4 = np.zeros([1, dims])
    x_final = np.zeros([dims + 1,n])
    it = 0
    while it < n:
        for i in range(dims):
            k1[0, i] = (f[i](x, t))
        for i in range(dims):
            k2[0, i] = (f[i](x + .5*h*k1, t + h/2.))
        for i in range(dims):
            k3[0, i] = (f[i](x + .5*h*k2, t + h/2.))
        for i in range(dims):
            k4[0, i] = (f[i](x + h*k3, t + h))
        x_next = x + ((h/6.) * (k1 + 2.*k2 + 2.*k3 + k4))
        for i in range(dims + 1):
            if i < dims:
                x_final[i , it] = x_next[0, i]
            else:
                x_final[i , it] = t
        x = x_next
        t = t+h
        it += 1

    if save_me:
        with open(path, 'w') as fs:
            fs.write(x_final)
            fs.close()

    return x_final

#Spatial poincare section given a function f,
#trajectory traj, norm of a plane you want to use for sectioning,
#and a point on the plane
def space_poincare(f, traj, norm, point):
    rows = len(traj)
    dims = rows - 1
    tol = .1

    poincare_section = []

    for i in range(len(traj[0]) - 1):
        x1 = traj[:, i]
        x2 = traj[:,i+1]
        d1 = np.dot((point - x1), norm)
        d2 = np.dot((point - x2), norm)

        if d1 < 0 and d2 > 0:
            poincare_section.append(x2)

    return poincare_section


#Spatial poincare that interpolates points
def space_poincare_inter(f, traj, norm, point):
    rows = len(traj)
    dims = rows - 1
    tol = .1

    poincare_section = []

    for i in range(len(traj[0]) - 1):
        x1 = traj[:, i]
        x2 = traj[:,i+1]
        d1 = np.dot((point - x1), norm)
        d2 = np.dot((point - x2), norm)

        if d1 < 0 and d2 > 0:
            b_min = 0
            b_max = .005
            x_new = x1
            count = 0
            while abs(d1) > tol and count < 1000:
                h = (b_max + b_min)/2.
                x_new = rk4_step(f, np.array([x1]), 0, h)[0]
                d1 = np.dot((point - x_new), norm)
                if d1 < 0:
                    b_min = h
                elif d1 > 0:
                    b_max = h
                count += 1
            poincare_section.append(x_new)

    return poincare_section

    