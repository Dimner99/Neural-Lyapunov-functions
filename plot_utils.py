import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

def plot_model_and_system(model, system, domain, args, num_points, plot_streamplot=True, figsize=(10, 8), 
                         cmap='viridis', contour_levels=50, alpha=0.7, streamplot_color='white',
                         density=1.0, linewidth=1.0, arrowsize=1.0, title=None,
                         show_colorbar=True, save_path=None, show_plot=True):
    """
    Plot the model evaluation as a contour plot and optionally overlay with the system's streamplot.
    
    Args:
        model: A function that takes points of shape (N, 2) and returns values of shape (N,).
        system: A function that takes points of shape (N, 2) and returns vectors of shape (N, 2).
        domain: A tuple of ((xmin, xmax), (ymin, ymax)) defining the plot domain.
        args: Parameters to pass to the system function.
        num_points: Number of points in each dimension for the grid.
        plot_streamplot: Boolean flag to include streamplot overlay.
        figsize: Figure size as a tuple (width, height).
        cmap: Colormap for the contour plot.
        contour_levels: Number of levels for the contour plot.
        alpha: Alpha transparency for the contour plot.
        streamplot_color: Color of the streamlines.
        density: Density of streamplot arrows.
        linewidth: Width of streamplot lines.
        arrowsize: Size of streamplot arrows.
        title: Title for the plot.
        show_colorbar: Whether to display the contour's colorbar (ignored if model is None).
        save_path: A file path to save the resulting figure, if provided.
        show_plot: If True, calls plt.show() at the end.
    
    Returns:
        fig, ax: The matplotlib figure and axis objects.
    """
    (xmin, xmax), (ymin, ymax) = domain
    
    # Use numpy.linspace for strictly equal spacing
    x = np.linspace(float(xmin), float(xmax), num_points)
    y = np.linspace(float(ymin), float(ymax), num_points)
    X, Y = np.meshgrid(x, y)
    
    # Ravel the meshgrid to create a JAX array of shape (num_points*num_points, 2)
    grid_points = jnp.stack([jnp.array(X.ravel()), jnp.array(Y.ravel())], axis=-1)
    
    if model is not None:
        # Forward pass through the model
        Z = model(grid_points).reshape(X.shape)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create contour plot
        contour = ax.contourf(X, Y, Z, levels=contour_levels, cmap=cmap, alpha=alpha)
        if show_colorbar:
            plt.colorbar(contour, ax=ax)
    else:
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
    
    if plot_streamplot:
        # Compute vector field for streamplot
        vector_field = system(grid_points, args)
        
        # Reshape to separate U, V components
        U = vector_field[:, 0].reshape(X.shape)
        V = vector_field[:, 1].reshape(Y.shape)
        
        # Convert to numpy for matplotlib streamplot
        # Use 1D coordinate arrays for x and y (not the meshgrid)
        x_np = np.array(x)
        y_np = np.array(y)
        U_np = np.array(U)
        V_np = np.array(V)
        
        # Create streamplot - use 1D arrays for coordinates
        ax.streamplot(x, y, U, V, density=density, linewidth=linewidth, 
                     arrowsize=arrowsize, color=streamplot_color)
    
    if title:
        ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    return fig, ax
#%%
