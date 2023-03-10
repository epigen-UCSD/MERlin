from matplotlib import pyplot as plt

from merlin.analysis.segment import FeatureSavingAnalysisTask
from merlin.plots._base import AbstractPlot


class SegmentationBoundaryPlot(AbstractPlot):
    def __init__(self, plot_task):
        super().__init__(plot_task)
        self.set_required_tasks({"segment_task": FeatureSavingAnalysisTask})

    def create_plot(self, **kwargs):
        feature_db = kwargs["tasks"]["segment_task"].get_feature_database()
        features = feature_db.read_features()

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        ax.set_aspect("equal", "datalim")

        if len(features) == 0:
            return fig

        z_position = 0
        if len(features[0].get_boundaries()) > 1:
            z_position = int(len(features[0].get_boundaries()) / 2)

        features_z = [feature.get_boundaries()[int(z_position)] for feature in features]
        features_z = [x for y in features_z for x in y]
        coords = [
            [feature.exterior.coords.xy[0].tolist(), feature.exterior.coords.xy[1].tolist()] for feature in features_z
        ]
        coords = [x for y in coords for x in y]
        plt.plot(*coords)

        plt.xlabel("X position (microns)")
        plt.ylabel("Y position (microns)")
        plt.title("Segmentation boundaries")
        return fig
