# Author: Leonardo Rossi Leão
# E-mail: leonardo.leao@cnpem.br

import glob
import os, shutil
from PIL import Image
from colour import Color
from datetime import datetime
from actions.math import Math
from alive_progress import alive_bar

# Data processing libraries
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev

# Matplotlib libraries
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText

"""
    Talvez eu use:

    # Formato o eixo de datetime
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d, %H:%M'))

    # Adiciona uma caixa com o horário da medição
    at = AnchoredText(
        dt.strftime("%d/%m/%Y, %H:%M:%S"), prop=dict(size=12), frameon=True, loc='upper right')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
"""


class Plot():

    @staticmethod
    def generate_gif(origin_folder: str, duration: int, delete_origin: bool = False) -> None:

        # Sort the frames based on the time that it was created
        files = glob.glob(f"{origin_folder}/*.png")
        files.sort(key=os.path.getmtime)
        frames = [Image.open(image) for image in files]

        # Set the first frame
        gif = frames[0]

        # Creating gif
        now = datetime.now()
        filename = f"hls_" + now.strftime("%Y-%m-%d_%H-%M") + ".gif" 
        gif.save(filename, format="GIF", append_images=frames, save_all=True,
                 duration=duration, loop=0)
        
        # Delete the origin folder if delete_origin is True
        if delete_origin:
            Plot.__remove_folder(origin_folder)

        print("Done:", filename)
        
    @staticmethod
    def dict_to_pandas(data_dict: dict) -> pd.DataFrame:

        # Re-organizing data of archiver
        convert = {}
        pvs = data_dict.keys()
        for pv in sorted(pvs, key=lambda pv: pv[16:18]):
            convert[pv] = data_dict[pv]["y"]
        convert["datetime"] =  data_dict[pv]["x"]
        
        # Creating the dataframe
        dataframe = pd.DataFrame.from_dict(convert)

        return dataframe.set_index("datetime")
    
    @staticmethod
    def smoothing(x: list, y: list, resolution: int) -> tuple:
        if len(x) > 3:
            smooth_x = np.linspace(x[0], x[-1], resolution)
            tck = splrep(x, list(y), s=0)
            smooth_y = splev(smooth_x, tck, der=0)

            return (smooth_x, smooth_y)
        else:
            print("Insufficient amount of points.")

    @staticmethod
    def diff(dataframe: pd.DataFrame) -> pd.DataFrame:
        df = dataframe.copy()
        for column in df.columns.values:
            df[column] -= df[column].iloc[0]
        return df

    @staticmethod
    def min_max(dataframe: pd.DataFrame) -> tuple:
        min = dataframe.min(numeric_only=True).min()
        max = dataframe.max(numeric_only=True).max()
        return min, max
    
    @staticmethod
    def datetime_box(datetime: datetime, axis) -> None:
        at = AnchoredText(
            datetime.strftime("%d/%m/%Y, %H:%M:%S"), 
            prop=dict(size=10), frameon=True, loc='upper left'
        )
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        axis.add_artist(at)

    @staticmethod
    def create_folder(str) -> str:
        now = datetime.now()
        folder = f"./hls_{str}_" + now.strftime("%Y-%m-%d_%H-%M")
        os.mkdir(folder)
        return folder
    
    @staticmethod
    def __remove_folder(folder: str) -> None:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        os.removedirs(folder)
    
class Plot2D(Plot):

    # Defining default figure parameters
    def __figure_ax(diff: bool, label: str = "Sirius Building Axes") -> plt.figure:

        figure = plt.figure(figsize=(10, 4))
        ax = figure.add_subplot(111)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$\Delta$Level [mm]" if diff else "Level [mm]")
        ax.grid(linestyle="--", color="#D3D3D3")
        return figure, ax
    
    # Setting colorbar
    def __colorbar(cmap, figure, dataframe) -> None:

        # creating ScalarMappable
        norm = mpl.colors.Normalize(vmin=0, vmax=4)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        
        # Formatting colorbar
        cb = figure.colorbar(sm, ticks=[0, 1, 2, 3, 4])
        df_datetimes = pd.to_datetime(dataframe.index.values)
        ticks = [
            df_datetimes[0].strftime("%b-%d, %H:%M"),
            df_datetimes[int(dataframe.shape[0]*0.25)].strftime("%b-%d, %H:%M"),
            df_datetimes[int(dataframe.shape[0]*0.5)].strftime("%b-%d, %H:%M"),
            df_datetimes[int(dataframe.shape[0]*0.75)].strftime("%b-%d, %H:%M"),
            df_datetimes[-1].strftime("%b-%d, %H:%M")
        ]
        cb.ax.set_yticklabels(ticks)

    @staticmethod
    def static_plot(level_measurements: dict, diff: bool = True) -> None:
        figure, ax = Plot2D.__figure_ax(diff=diff, label="Time")
        for pv in level_measurements.keys():
            x = level_measurements[pv]["x"]
            y = np.array(level_measurements[pv]["y"])

            # Apply difference
            y -= y[0] if diff else 0 

            ax.plot(x, y)
        
        # Format chart style
        ax.set_xlim(x[0], x[-1])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d, %H:%M'))
        figure.tight_layout()
        plt.show()

    @staticmethod
    def static_axis_plot(level_measurements: dict, diff: bool = True) -> None:

        # Converting data to pandas dataframe
        dataframe = Plot2D.dict_to_pandas(level_measurements)
        if diff:
            dataframe = Plot2D.diff(dataframe)
        axes = [int(pv[16:18]) for pv in dataframe.columns.values]
        
        figure, ax = Plot2D.__figure_ax(diff=diff)
        ax.set_xlim(axes[0], axes[-1])

        # Defining a color gradient
        cmap = plt.get_cmap('viridis', dataframe.shape[0])

        # Loop through dataframe rows
        with alive_bar(dataframe.shape[0], title="Generating 2D static plot by axis") as bar:

            for row in range(dataframe.shape[0]):
                axes_spline, level_spline = Plot2D.smoothing(axes, dataframe.iloc[row].values, 180)
                ax.plot(axes_spline, level_spline, c=cmap(row))
                bar()

        Plot2D.__colorbar(cmap, figure, dataframe)
        figure.tight_layout()
        plt.show()

    @staticmethod
    def dynamic_axis_plot(level_measurements: dict, save_all_figures: bool = True) -> None:

        # Converting data to pandas dataframe
        dataframe = Plot2D.diff(Plot2D.dict_to_pandas(level_measurements))
        axes = [int(pv[16:18]) for pv in dataframe.columns.values]

        # Loop through dataframe rows
        with alive_bar(dataframe.shape[0], title="Generating 2D dynamic plot") as bar:

            # Creating folder to save figures
            folder = Plot2D.create_folder("2d_dynamic")

            for row in range(dataframe.shape[0]):
                figure, ax = Plot2D.__figure_ax(diff=True)
                Plot2D.datetime_box(dataframe.index[row], ax)

                # Axes limits
                ax.set_xlim(axes[0], axes[-1])
                min_y, max_y = Plot2D.min_max(dataframe)
                ax.set_ylim(min_y*1.1, max_y*1.1)

                # Smoothing the curve
                axes_spline, level_spline = Plot2D.smoothing(axes, dataframe.iloc[row].values, 180)

                # Plotting data
                ax.scatter(axes, dataframe.iloc[row].values)
                ax.plot(axes_spline, level_spline)
                figure.tight_layout()

                # Saving figure
                plt.savefig(f"{folder}/hls_{row}.png")
                plt.close()
                bar()

        # Generating gif and saving
        delete_folder = not save_all_figures
        Plot2D.generate_gif(folder, duration=70, delete_origin=delete_folder)

class Plot3D(Plot):

    # Global parameters
    RESOLUTION = 500
    RADIUS = np.linspace(350, 500, RESOLUTION)
    ANGLES = np.linspace(0, 360, RESOLUTION)

    # Mapping Sirius Radiological Shield in XY plane
    X = np.outer(RADIUS, np.cos(ANGLES * np.pi/180))
    Y = np.outer(RADIUS, np.sin(ANGLES * np.pi/180))

    # Defining the cut and fill line
    x_ini, y_ini = Math.polar_to_rectangular(RADIUS[-1], Math.get_angle(60, 2))
    x_end, y_end = Math.polar_to_rectangular(RADIUS[-1], Math.get_angle(60, 32))
    CUT_AND_FILL = {
        "x_ini": x_ini,
        "y_ini": y_ini,
        "x_end": x_end,
        "y_end": y_end,
    }

    @staticmethod
    def dynamic_plot(level_measurements: dict, save_all_figures: bool = True) -> None:

        # Converting data to pandas dataframe
        dataframe = Plot3D.diff(Plot3D.dict_to_pandas(level_measurements))
        axes = [int(pv[16:18]) for pv in dataframe.columns.values]

        # Loop through dataframe rows
        with alive_bar(dataframe.shape[0], title='Generating 3D dynamic plot') as bar:

            # Creating folder to save figures
            folder = Plot3D.create_folder("3d_dynamic")

            for row in range(dataframe.shape[0]):

                # Generating Z level
                level = dataframe.iloc[row].values
                AXES, z = Plot3D.smoothing(axes, level, 500)
                Z = np.outer(z, np.ones(z.shape)).T

                # Plotting data
                figure = plt.figure()
                ax = figure.add_subplot(111, projection="3d")
                z_min, z_max = Plot3D.min_max(dataframe)
                surface = ax.plot_surface(Plot3D.X, Plot3D.Y, Z, cmap="viridis",
                                        linewidth=0, antialiased=False, vmin=z_min, vmax=z_max)
                ax.set_zlim(z_min*1.1, z_max*1.1)

                # Cut and fill line
                try:
                    Plot3D.CUT_AND_FILL["z_ini"] = z[np.where(AXES.astype(int) == 2)[0][0]]
                    Plot3D.CUT_AND_FILL["z_end"] = z[np.where(AXES.astype(int) == 32)[0][0]]
                    ax.plot((Plot3D.CUT_AND_FILL["x_ini"], Plot3D.CUT_AND_FILL["x_end"]), 
                            (Plot3D.CUT_AND_FILL["y_ini"], Plot3D.CUT_AND_FILL["y_end"]), 
                            (Plot3D.CUT_AND_FILL["z_ini"], Plot3D.CUT_AND_FILL["z_end"]))
                except: pass

                # Box with datetime and colorbar
                Plot3D.datetime_box(dataframe.index[row], ax)
                cbar = figure.colorbar(surface, shrink=0.5, aspect=10, pad=0.15)
                cbar.set_label(r"$\Delta$Level [mm]", rotation=270, labelpad=15)

                # Saving figure
                plt.savefig(f"{folder}/hls_{row}.png")
                plt.close()
                bar()

        # Generating gif and saving
        delete_folder = not save_all_figures
        Plot3D.generate_gif(folder, duration=70, delete_origin=delete_folder)