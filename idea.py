

class Geometry:

    # parameters used in plot_shape
    plot_params : Dict['str', Any] = {'linewidth':2, 'color':'red'}

    def plot(x=None,y=None,z=None,ax=None):
        """Defines how the cross section gets plotted with matplotlib."""
        shapes = get_intersections(x=x,y=y,z=z)
        for shape in shapes:
            ax = self.plot_shape(shape, ax)
        return ax

    def plot_shape(self, shape, ax):
        """Defines how a shapely shape of this class gets added to matplotlib axes."""
        ax.add_patch(shape, **self.plot_params)
        return ax
