#%%
import os
import time as time
from functools import partial
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from plot_utils import plot_model_and_system

#%%
backend = jax.devices()[0].platform
device_jax = jax.devices()[0]
jax.config.update("jax_platform_name", backend)

# @partial(jax.jit, static_argnames=("args"),backend=backend)
@partial(jax.vmap,in_axes=(0,None))
def system(x, args):
    # (mu,) = args
    (mu,) = args
    x1, x2 = x[0], x[1]
    #! DEFINE THE DESIRED 2D SYSTEM
    # SYSTEM: x1' = -x2,  x2' = x1 - mu*(1 - x1**2)*x2
    return jnp.array([
        -x2,
        x1 - mu * (1 - x1**2) * x2
    ])
    # return jnp.array([x2, -(x1**5-x1**3+g*x1)-0.5*x2])



# @partial(jax.jit, backend=backend)
@jax.vmap
def omega(x: jax.Array):
    return jnp.sum(x**2)

# @partial(jax.jit, backend=backend)
def psi(x: jax.Array):
    return 0.1*(1+x)


class Net(nnx.Module):
    def __init__(self,din:int,layers_width:list[int],dout:int=1,*,rngs: nnx.Rngs):
        self.din = din
        self.dout = dout
        self.rngs = rngs
        self.layers_width = layers_width
        self.layers = [nnx.Linear(in_features=din, out_features=layers_width[0], rngs=self.rngs)]
        for i in range(1,len(layers_width)):
            self.layers.append(nnx.Linear(in_features=layers_width[i-1], out_features=layers_width[i], rngs=self.rngs))
        self.out = nnx.Linear(in_features=layers_width[-1], out_features=dout, rngs=self.rngs)        

    # @nnx.jit
    def __call__(self,x: jax.Array):  #&# This will be jitted once during training because we use it 
        for layer in self.layers:     #&# for the data points only 
            x = layer(x)
            x = nnx.tanh(x)
        return self.out(x).squeeze()  
    
    # @nnx.jit                              #&# This will be jitted once during training because we use it
    @partial(nnx.vmap,in_axes=(None,0))   #&# for the collocations points only
    def value_and_gradient(self, x):      
        val_,grad_ = nnx.value_and_grad(self.__call__)(x)
        return val_,grad_
    
# @jax.jit
def jacobian_fn(model):
    return nnx.vmap(nnx.grad(model))


# @jax.jit
def dot_vector(x, y):
    return jnp.sum(x * y, axis=1)


#%%

@partial(nnx.jit, static_argnames=("system", "omega", "psi", "args"))
def Zubov_training(col_points:jax.Array,
                   data_points:tuple[jax.Array, jax.Array],
                   system:callable, model:Net, 
                   omega:callable,
                   psi:callable,
                   args:tuple,
                   optimizer:optax.GradientTransformation) -> jax.Array:
    """Physics-informed neural network training using Zubov's method.
    The PDE to be solve is DW · f = -ω ψ(W)(1 - W) where W is the neural network function and DW the 
    gradient of W with respect to x. The loss is the sum of the PDE loss and the data loss.
    The PDE loss is computed at the collocation points and the data loss at the data points.
    We also impose the boundary loss by setting W(0)=0. The boundary loss is computed for each batch.

    Args:
        col_points (jax.Array): the points wherer the loss of the PDE is computed 
        data_points (tuple[jax.Array, jax.Array]): the points and values of the data loss
        system (callable): The dynamical system in format: system(x,args):x1,x2=x[0],x[1]...return jnp.array([f1(x),f2(x)])
        model (Net): The neural network model, an nnx.Module child
        omega (callable): ω(x) (see above equation)
        psi (callable): ψ(W(x)) (see above equation)
        args (tuple): args for the system function
        optimizer (optax.GradientTransformation): optax optimizer, it must have been initialized with the model
    """
    
    
    def Zubov_data_loss(model:nnx.Module):
        x_zeros=jnp.zeros(2)
        COLLOC_MODEL,GRAD_MODEL= model.value_and_gradient(col_points) # type: ignore
        COLLOC_SYS=system(col_points,args)
        COLLOC_OMEGA=omega(col_points)
        ZEROS_MODEL=model(x_zeros) # type: ignore
        DATA_MODEL=model(data_points[0]) # type: ignore

        
        zubov_loss=(dot_vector(GRAD_MODEL,COLLOC_SYS)+psi(COLLOC_MODEL)*COLLOC_OMEGA*(1-COLLOC_MODEL))**2 + ZEROS_MODEL**2
        zubov_loss_batch=jnp.mean(zubov_loss)
        data_loss=(DATA_MODEL-data_points[1])**2
        data_loss_batch=jnp.mean(data_loss) 
        loss_batch=data_loss_batch+zubov_loss_batch
        return loss_batch
    
    loss, grads = nnx.value_and_grad(Zubov_data_loss)(model)
    optimizer.update(model=model, grads=grads) # type: ignore
    return loss


#%%
num_points=320_000
batch_size=32
batch_size_len=num_points//batch_size
minx=-3
maxx=3
miny=-10
maxy=10
domain=((minx,maxx),(miny,maxy))

model1=Net(2,[10,10,10],dout=1,rngs=nnx.Rngs(1100))
optimizer1=nnx.Optimizer(model1, optax.adam(learning_rate=1e-3),wrt=nnx.Param)
args=(5.0,)
# args=(0.125,)
#%%
########### FURTHER OPTIMIZATION THROUGH jax.fori_loop FOR THE TRAINING LOOP ############
#^############# SPEEDUP 2X for CPU and 10X for GPU compared to the for loop  #^#############

model3=Net(2,[30,30,30],dout=1,rngs=nnx.Rngs(1020))
optimizer3=nnx.Optimizer(model3, optax.adam(learning_rate=1e-3),wrt=nnx.Param)
data_points=RESULTS

#%%
def train_model(model:Net,optimizer:nnx.Optimizer, num_points:int,batch_size:int,
                data_points:tuple[jax.Array, jax.Array]
                , system:callable, omega:callable, psi:callable,
                domain:tuple[tuple], args:tuple, key:jax.random.key) -> tuple[Net,nnx.Optimizer,jax.Array]:
    
    """
    Train the model using Zubov's method. The training loop is implemented using jax.fori_loop for speedup.
    num_iterations is the number of iterations to train the model.Each iteration uses a new batch of 
    collocation points, produces from new keys spltted from the key.
    
    Args:
        model (Net): The neural network model, an nnx.Module child
        optimizer (nnx.Optimizer): The optimizer for the model
        num_points (int): The number of points to train the model
        batch_size (int): The number of points in each batch
        data_points (tuple[jax.Array, jax.Array]): The data points and values for the loss function
        system (callable): The dynamical system in format: system(x,args):x1,x2=x[0],x[1]...return jnp.array([f1(x),f2(x)])
        omega (callable): ω(x) (see above equation)
        psi (callable): ψ(W(x)) (see above equation)
        domain (tuple[tuple]): The domain of the system, in the format ((minx,maxx),(miny,maxy))
        args (tuple): args for the system function
        key (jax.random.key): The random key for jax
    """
    
    num_iterations = num_points // batch_size

    @nnx.jit
    def body_fn(i, state):
        modell, optimizer, loss, current_key = state
        current_key, key1, key2 = jax.random.split(current_key, num=3)
        x = jax.random.uniform(key1, (batch_size, 1), minval=domain[0][0], maxval=domain[0][1])
        y = jax.random.uniform(key2, (batch_size, 1), minval=domain[1][0], maxval=domain[1][1])
        xy = jnp.concatenate((x, y), axis=1)
        loss = loss.at[i].set(Zubov_training(xy, data_points, system, modell, omega, psi, args, optimizer))
        return modell, optimizer, loss, current_key
    
    loss=jnp.empty((num_iterations,),dtype=jnp.float32)
    
    model_final, optimizer_final, loss_final, _ = nnx.fori_loop(0, num_iterations, body_fn, 
                                                              (model, optimizer, loss, key), unroll=False)
    return model_final, optimizer_final, loss_final
#%%
for _ in range(1):
    time1 = time.time()
    model3, optimizer3, loss3 = train_model(model3, optimizer3, num_points, batch_size, data_points, 
                                            system, omega, psi, domain, args, jax.random.PRNGKey(0))
    x = jax.random.uniform(jax.random.key(1), (num_points//10, 1), minval=minx, maxval=maxx)
    y = jax.random.uniform(jax.random.key(4), (num_points//10, 1), minval=miny, maxval=maxy)
    xy = jnp.concatenate((x, y), axis=1)
    results = model3(xy)
    print(results.shape)
    plt.figure()
    plt.scatter(xy[:,0], xy[:,1], c=results, s=0.1, cmap="viridis", vmin=jnp.min(results), vmax=jnp.max(results))
    plt.colorbar()  
    plt.title("Zubov's method")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()
    time2 = time.time()
    print("Time taken for training with jax.fori_loop:", time2 - time1)
#%%

plot_model_and_system(model3, system, domain,args,1000,streamplot_color="black",density=1,linewidth=0.5,
                      contour_levels=20,cmap="viridis",alpha=0.9,figsize=(12,10),title="Zubov's method")
# %%
plot_model_and_system(None, system, domain,args,1000,streamplot_color="black",density=1.3,linewidth=0.5,
                      contour_levels=20,cmap="viridis",alpha=0.9,figsize=(12,10),title="Zubov's method")
# %%
