from merlin.core import analysistask
import pandas

class AggregateData(analysistask.AnalysisTask):
    """
    A metaanalysis task that aggregates data from multiple datasets
    """
    def __init__(self, metaDataSet, parameters=None, analysisName=None):
        super().__init__(metaDataSet, parameters, analysisName)

        if 'overwrite' not in self.parameters:
            self.parameters['overwrite'] = False

        self.metaDataSet = metaDataSet

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 100

    def get_dependencies(self):
        dep = [v for k, v in self.parameters.items() if 'task' in k]
        if len(dep) > 1:
            print('This task can only aggregate one analysis task')
        else:
            dependencies = []
            allDataSets = self.metaDataSet.load_datasets()
            for k, v in allDataSets.items():
                dependencies.append(v.load_analysis_task(dep[0]))
        return dependencies

    def load_aggregated_data(self, analysisName: str, **kwargs):
        return self.metaDataSet.load_dataframe_from_csv(analysisName,
                                                        self, **kwargs)

    def _run_analysis(self):
        allAnalyses = []
        deps = self.get_dependencies()
        for d in deps:
            try:
                tempData = d.return_exported_data()
                tempData['dataset'] = d.dataSet.dataSetName
                allAnalyses.append(tempData)
            except FileNotFoundError:
                print('{} result not found for dataset {}'.format(
                    d.analysisName, d.dataSet.dataSetName))
        if len(allAnalyses) == len(self.metaDataSet.dataSets):
            combinedAnalysis = pandas.concat(allAnalyses, 0)
            self.metaDataSet.save_dataframe_to_csv(combinedAnalysis,
                                                   [v for k, v in
                                                    self.parameters.items()
                                                    if 'task' in k][0],
                                                   self)
