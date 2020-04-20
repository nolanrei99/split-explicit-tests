using LinearAlgebra
using FFTW
using Random
using PyPlot

macro PRINT(mat)
    return :(show(Base.stdout, "text/plain", $mat))
end

function imex_cn(x0, A, f, dt, t0, T)
    N = size(A)[1]
    dt2 = 0.5*dt
    II = Matrix{Float64}(I,N,N)     # Identity
    F = II - dt2*A
    F = factorize(F)
    B = II + dt2*A
    x2 = zeros(N)
    rhs = zeros(N)
    t = t0+0.0
    x = 1.0*x0

    # Plotting stuff
    n = convert(Int64,(T-t0)/dt)
    xs = zeros(n+1)
    ts = zeros(n+1)
    xs[1] = x0[1]
    ts[1] = t0
    i = 1

    while t <= T && i <= n
        # Project to center of timestep
        x2 .= x .+ dt2.*(A*x.+f(x))
        
        # Crank-Nicolson step
        rhs .= B*x .+ dt*f(x2)
        x .= F\rhs

        t += dt
        i += 1
        xs[i] = x[1]
        ts[i] = t
    end
    return xs,ts
end

function splex(x0, ffast, fslow, divider, dt, dt_, t0, T)
    # Split-explicit scheme for differential equations
    # 2nd-order, uses Midpoint (RK2) method
    # ffast is the portion of the function that operates on the fast variables
    # fslow is for the slow variables
    # divider is the index separating the slow variables from the fast variables
    # dt is the timestep for the slow variables, dt_ is the timestep for the fast variables
    
    N = length(x0)
    t = t0+0.0
    t_ = t+0.0
    x = 1.0*x0
    x2 = zeros(N)
    x_ = view(x, divider:N)
    x2_ = zeros(N)
    dt2 = 0.5*dt
    dt_2 = 0.5*dt_
    d_ = dt_/dt

    # Plotting stuff
    n = convert(Int64,(T-t0)/dt)
    xs = zeros(n+1)
    ts = zeros(n+1)
    xs[1] = x0[1]
    ts[1] = t0
    i = 1

    # Slow loop
    while t <= T && i <= n
        t += dt
        # Fast loop
        x2_[1:divider-1] .= x[1:divider-1]
        x2[divider:N] .= 0.0
        while t_ <= t
            t_ += dt_
            x2_[divider:N] .= x_ .+ dt_2.*ffast(x) # needs to view slow components as well as fast
            x_ .+= dt_*ffast(x2_)
            x2[divider:N] .+= d_*x_
        end
        # Project to midpoint
        x2[1:divider-1] .= x[1:divider-1]
        x2[1:divider-1] .+= dt2.*fslow(x2)

        # Use midpoint to extrapolate to endpoint
        x[1:divider-1] .+= dt*fslow(x2)

        i += 1
        xs[i] = x[1]
        ts[i] = t
    end
    return xs,ts
end

function splex_ctr(x0, ffast, fslow, divider, dt, dt_, t0, T)
    # Split-explicit scheme for differential equations
    # 2nd-order, uses Midpoint (RK2) method
    # ffast is the portion of the function that operates on the fast variables
    # fslow is for the slow variables
    # divider is the index separating the slow variables from the fast variables
    # dt is the timestep for the slow variables, dt_ is the timestep for the fast variables
    # INTEGRATES FAST TERMS OVER t_{n-1/2} to t_{n+1/2} INSTEAD OF t_{n-1} TO t_n
    
    N = length(x0)
    t = t0+0.0
    t_ = t+0.0
    x = 1.0*x0
    x2 = zeros(N)
    x_ = view(x, divider:N)
    x2_ = zeros(N)
    dt2 = 0.5*dt
    dt_2 = 0.5*dt_
    d_ = dt_/dt

    # Plotting stuff
    n = convert(Int64,(T-t0)/dt)
    xs = zeros(n+1)
    ts = zeros(n+1)
    xs[1] = x0[1]
    ts[1] = t0
    i = 1

    # Initialize the offsetting
    x2_[1:divider-1] .= x[1:divider-1]
    while t_ <= t+0.5*dt
        t_ += dt_
        x2_[divider:N] .= x_ .+ dt_2.*ffast(x) # needs to view slow components as well as fast
        x_ .+= dt_*ffast(x2_)
    end

    # Slow loop
    while t <= T && i <= n
        t += dt
        # Fast loop
        x2_[1:divider-1] .= x[1:divider-1]
        x2[divider:N] .= 0.0
        while t_ <= t+0.5*dt
            t_ += dt_
            x2_[divider:N] .= x_ .+ dt_2.*ffast(x) # needs to view slow components as well as fast
            x_ .+= dt_*ffast(x2_)
            x2[divider:N] .+= d_*x_
        end
        # Project to midpoint
        x2[1:divider-1] .= x[1:divider-1]
        x2[1:divider-1] .+= dt2.*fslow(x2)

        # Use midpoint to extrapolate to endpoint
        x[1:divider-1] .+= dt*fslow(x2)

        i += 1
        xs[i] = x[1]
        ts[i] = t
    end
    return xs,ts
end

eps = 0.1
b = 60.0
A = zeros(5,5)
A[4,5] = -1.0/eps
A[5,4] = 1.0/eps
function lokri(x)
    xdot = zeros(5)
    xdot[1] = -x[2]*x[3] + eps*b*x[2]*x[5]
    xdot[2] = x[1]*x[3] - eps*b*x[1]*x[5]
    xdot[3] = -x[1]*x[2]
    xdot[4] = 0.0
    xdot[5] = b/eps*x[1]*x[2]
    return xdot
end
function lokri_fast(x)
    xdot = zeros(2)
    xdot[1] = -1.0/eps*x[5]
    xdot[2] = 1.0/eps*x[4] + b/eps*x[1]*x[2]
    return xdot
end
function lokri_slow(x)
    xdot = zeros(3)
    xdot[1] = -x[2]*x[3] + eps*b*x[2]*x[5]
    xdot[2] = x[1]*x[3] - eps*b*x[1]*x[5]
    xdot[3] = -x[1]*x[2]
    return xdot
end

t0 = 0.0
T = 600.0
dt = 0.01
Random.seed!(42)
x0 = 0.05*randn(5)

#xs,ts = imex_cn(x0,A,lokri,dt,t0,T)
dt_ = 0.001
xs,ts = splex(x0, lokri_fast, lokri_slow, 4, dt, dt_, t0, T)
plot(ts,xs)
show()
