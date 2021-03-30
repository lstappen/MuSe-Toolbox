import numpy as np
import matplotlib.markers as markers
import matplotlib.pyplot as plt
import pandas as pd


class RadarChart:

    """
        RadarChart aka SpiderChart to profile the resulting clusters
    """

    def __init__(self, labels):
        """ Initialize to necessary data for the RadarChar. 
            Main logic: Calculate the angles for the number of labels.

        Args:
            labels ([string]): The labels of the RadarChart. 
        """
        self.fontsize = 12

        self.labels = [f'({i+1}) {label}' for i, label in enumerate(labels)]
        self.labels = [label.replace('_arousal', '').replace('_valence', '') for label in self.labels]

        N = len(self.labels)

        self.fig, self.ax = plt.subplots(figsize=(10, 7), subplot_kw=dict(polar=True))
        # self.fig, self.ax = plt.subplots(figsize=(15, 7), subplot_kw=dict(polar=True), nrows=1, ncols=2, gridspec_kw={'width_ratios': [3, 1]})

        # Split the circle into even parts and save the angles.
        self.angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()

        # complete the loop
        self.angles += self.angles[:1]

        # Fix axis to go in the right order and start at 12 o'clock.
        self.ax.set_theta_offset(np.pi / 2)
        self.ax.set_theta_direction(-1)

        # Draw axis lines for each angle and label.
        self.ax.set_thetagrids(np.degrees(self.angles[:-1]), self.labels, fontsize=self.fontsize)

        # Go through labels and adjust alignment based on where
        # it is in the circle.
        for label, angle in zip(self.ax.get_xticklabels(), self.angles):
            if angle in (0, np.pi):
                label.set_horizontalalignment('center')
            elif 0 < angle < np.pi:
                label.set_horizontalalignment('left')
            else:
                label.set_horizontalalignment('right')

        self.bottom, self.top = self.ax.set_ylim(-1, 2.5)  # (-1, 2.75)

        # Set position of y-labels (0-100) to be in the middle
        # of the first two axes.
        self.ax.set_rlabel_position(180 / N)

        # Styling
        self.ax.tick_params(colors='#222222')
        self.ax.tick_params(axis='y', labelsize=12)
        self.ax.grid(color='#AAAAAA')
        self.ax.spines['polar'].set_color('#222222')
        self.ax.set_facecolor('#FAFAFA')

        # TODO fix
        '''
        ax = self.fig.add_subplot(1, 1, 1)
        ax.set_xlim(1, 3)
        ax.set_ylim(1, 3)
        ax.axis('off')
        arrow = markers.MarkerStyle(marker=r'$\circlearrowright$')
        arrow._transform = arrow.get_transform().rotate_deg(90)
        ax.plot([2.5], [2.5], marker=arrow, ms=50)
        '''

    def add_to_radar(self, label, color, data):
        """ Adds a data point to the RadarChart

        Args:
            label (string): The name
            color (color): The color of the label
            data ([float]): The data point
        """
        # complete the loop
        data += data[:1]

        self.ax.plot(self.angles, data, color=color, linewidth=1, label=label)
        self.ax.fill(self.angles, data, color=color, alpha=0.25)

        if max(data) > self.top:
            self.bottom, self.top = self.ax.set_ylim(top=max(data) + 0.5)
        if min(data) < self.bottom:
            self.bottom, self.top = self.ax.set_ylim(bottom=min(data) - 0.5)

    def show(self, title, show, export_dir, filename="Radar Chart"):
        """Show and save the plot

        Args:
            title (string): The name of the plot
            show (bool): If true, show the plot
            export_dir (str): If not empty, save plot as svg in export_dir
            filename (str): Filename to save the plot as (without extension)
        """
        # Add title.
        self.ax.set_title(title, y=1.08, fontsize=self.fontsize + 2)
        # Add a legend as well.
        #self.ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.3, 0.83), fontsize=self.fontsize)
        self.fig.tight_layout()
        if show: 
            plt.show()
        # save the figure
        if export_dir:
            self.fig.savefig(f"{export_dir}/{filename}.png", format="png")
        plt.close()
