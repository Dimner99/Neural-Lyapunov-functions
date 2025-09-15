#%%
import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt
import optimistix as optx
import equinox
import numpy as np
from jax.numpy import trapezoid
from functools import partial
import time
# import torch
# %%

backend = jax.devices()[0].platform
device_jax = jax.devices()[0]
jax.config.update("jax_platform_name", backend)
# device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##########################################
# 1. Define the ODE: Reverse Van der Pol
##########################################
def reverse_vdp(t, x, args):
    (mu,) = args
    x1, x2 = x
    # x1' = -x2,  x2' = x1 - mu*(1 - x1**2)*x2
    return jnp.array([
        -x2,
        x1 - mu * (1 - x1**2) * x2
    ])

minx=-3
maxx=3
miny=-10
maxy=10
domain=((minx,maxx),(miny,maxy))
maxnorm = jnp.sqrt(maxx**2 + maxy**2)
g=0.125
mu=5.0
delta=jnp.pi/3
args=(mu,)


##########################################ss
# 2. Create a steady-state event
##########################################
# This condition returns a value that becomes less than zero once ||f(t,y)|| < atol + rtol * ||y||
near_origin_equil = diffrax.steady_state_event(
    rtol=1e-6,  # Make this smaller for more precise equilibrium detection
    atol=1e-6,  # Make this smaller for more precise equilibrium detection
    norm=lambda z: jnp.linalg.norm(z, ord=2)
)
def non_origin_equilibrium_event(t, y, args, **kwargs):
    # Compute derivatives at current point
    dy_dt = reverse_vdp(t, y, args)
    # Get the norm of derivatives
    derivative_norm = jnp.linalg.norm(dy_dt, ord=2)
    
    # Check if point is near origin
    distance_from_origin = jnp.linalg.norm(y, ord=2)
    
    # Only return negative value when:
    # 1. Derivative is small enough (system not changing much) AND
    # 2. Distance from origin is NOT small (not at the trivial equilibrium)
    
    # Thresholds
    derivative_threshold = 1e-4
    origin_threshold = 1  # Points closer than this to origin are excluded
    
    # Check if derivatives are small (equilibrium condition)
    is_equilibrium = derivative_norm < derivative_threshold
    # Check if point is far enough from origin
    is_away_from_origin = distance_from_origin > origin_threshold
    
    # Return negative value only when both conditions are met
    return jax.lax.cond(
        is_equilibrium & is_away_from_origin,
        lambda _: True,  # If both conditions are met, return negative (trigger event)
        lambda _: False,  # Otherwise, return positive (don't trigger event)
        operand=None
    )

# Add a condition to detect if values have grown too large (divergence)
def overflow_event(t, y, args, **kwargs):
    # Check if any value has exceeded our threshold
    max_allowed = 1e5 # Maximum allowed value: 10^5
    max_value = jnp.max(jnp.abs(y))
    # Return negative value when the max value exceeds our threshold
    return max_allowed - max_value

# Use both the equilibrium event and the overflow event
steady_event = diffrax.Event(cond_fn={0: near_origin_equil,1: overflow_event,2:non_origin_equilibrium_event})

controller = diffrax.PIDController(rtol=1e-7, atol=1e-7)

system = diffrax.ODETerm(reverse_vdp)
solver = diffrax.Dopri8()



@partial(jax.jit, backend=backend)
@jax.vmap
def omega(x: jax.Array):
    return jnp.sum(x**2)




def batched_solve(num_points:int, batch_size:int, t0:jax.Array, t1:jax.Array,
                  domain:tuple[tuple],
                  system:diffrax.ODETerm,
                  solver:diffrax.AbstractSolver,
                  controller:diffrax.PIDController,
                  term_cond:diffrax.Event,
                  dti:float,
                  max_steps:int,
                  args:tuple,
                  omega:callable,
                  beta:callable,
                  key:jax.random.key) -> tuple[jax.Array,jax.Array]:

    """
    The event condition must have the following format:
    diffrax.Event(cond_fn={0: Valid_fn,1: Invalid_fn1,2: Invalid_fn2,...})
    where the first function is the valid function the points of which must be integrated across the trajectories
    The other events will be substituted with 1 without integration.
    One good approach for the most dynamical systems is:
    
    diffrax.Event(cond_fn={0: near_origin_equil,1: overflow_event,2:non_origin_equilibrium_event})
    
    where near_origin is the function that triggers a stop if the solution is very close to the origin
    which is equilibrium point.
    
    The overflow_event is the function that triggers a stop if the solution
    is too far from the origin, goes to infinity or diverges in general to a far away positive invariant set.
    
    The non_origin_equilibrium_event is the function that triggers a stop if the solution is close to an 
    equilibrium point that is not the origin.
    
    Note: t1 is recommended to be jnp.inf, the events will handle the time stop.
    Note2: The beta function is used inside the integration function to transform the results.
    Good options tanh(0.1*x) or 1-jnp.exp(-0.1*x)
    """
    step = num_points // batch_size

    @partial(jax.jit, backend=backend)
    @partial(jax.vmap, in_axes=(0,0))
    def integration(ys:jax.Array, ts:jax.Array) -> tuple[jax.Array,jax.Array,jax.Array]:
        yssq = omega(ys)
        is_valid = ~jnp.isinf(ts)
        integral = trapezoid(
            jnp.where(is_valid, yssq, 0.0),
            jnp.where(is_valid, ts, 0.0)
        )
        result = beta(integral)
        return result, ys[0,0], ys[0,1]

    @partial(jax.jit, backend=backend)
    @partial(jax.vmap, in_axes=(0,))
    def solver_(y0:jax.Array) -> tuple[jax.Array,jax.Array,jax.Array]:
        sol = diffrax.diffeqsolve(
            terms=system,
            solver=solver,
            t0=t0,
            t1=t1,
            dt0=dti,
            y0=y0,
            args=args,
            event=term_cond,
            max_steps=max_steps,
            saveat=diffrax.SaveAt(t0=True, steps=True),
            stepsize_controller=controller,
            throw=False)
        return sol.ys, sol.ts, sol.event_mask[0]

    def scan_fn(carry, _):
        key = carry
        key, key1, key2 = jax.random.split(key, 3)
    
        # sample init conditions
        x = jax.random.uniform(key1, (batch_size, 1), minval=domain[0][0], maxval=domain[0][1])
        y = jax.random.uniform(key2, (batch_size, 1), minval=domain[1][0], maxval=domain[1][1])
        init = jnp.concatenate((x, y), axis=1)
    
        # solve ODE
        ys, ts, mask = solver_(init)   # shapes: (batch_size, steps, dim), (batch_size, steps), (batch_size,)
    
        # --- Instead of slicing with mask ---
        # pick first step always (for consistency)
        y0_x1 = ys[:, 0, 0]
        y0_x2 = ys[:, 0, 1]
    
        # do integration for every trajectory
        int_results, _, _ = integration(ys, ts)
    
        # select either:
        #   - integrated value (if mask == True)
        #   - 1.0 as "fallback" (if mask == False)
        r_batch = jnp.where(mask, int_results, 1.0)
        x_batch = y0_x1
        y_batch = y0_x2
    
        return key, (x_batch, y_batch, r_batch)


    # run scan
    key, (xs, ys, rs) = jax.lax.scan(scan_fn, key, None, length=step)

    # flatten to (num_points,)
    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)
    rs = rs.reshape(-1)

    # match original return format
    coords = jnp.concatenate((xs, ys), axis=1)
    return coords.block_until_ready(), rs.block_until_ready()

    
#%%

NUM_POINTS=300000
Batch_size=30000
time1=time.time()
f1=lambda x: 1-jnp.exp(-0.1*x)
f2=lambda x: jnp.tanh(0.1*x)
for _ in range(1):
    RESULTS = batched_solve(NUM_POINTS, batch_size=Batch_size, t0=0.0, t1=jnp.inf,
                    domain=domain,
                    system=system,
                    solver=solver,
                    controller=controller,
                    term_cond=steady_event,
                    dti=0.1,
                    max_steps=1000,
                    args=args,
                    omega=omega,
                    beta=f2,
                    key=jax.random.key(0))
time2=time.time()
print("Time taken for {} points: {:.4f} seconds".format(NUM_POINTS, time2 - time1))


#%%
plt.figure()
plt.scatter(RESULTS[0][:,0], RESULTS[0][:,1], c=RESULTS[1], s=1, cmap="jet", vmin=jnp.min(RESULTS[1]), vmax=jnp.max(RESULTS[1]))
plt.colorbar()
plt.show()
#%%
