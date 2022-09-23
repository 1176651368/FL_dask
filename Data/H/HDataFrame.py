import dask.dataframe as df
from dask.dataframe.core import DataFrame


class HDataFrame:
    def __init__(self, data:DataFrame,role='guest'):
        self.org_data = data
        self.idx = self.org_data.columns[0]
        self.role = role
        self.feature_name, self.label_name = self.get_col()
        self.feature, self.label = self.get_feature_label()

    def get_col(self):
        index = 2 if self.role == 'guest' else 1
        feature_name = self.org_data.columns[index:]
        label_name = self.org_data.columns[index-1:index] if self.role == 'guest' else None
        return feature_name, label_name

    def get_feature_label(self):
        index = 2 if self.role == 'guest' else 1
        feature = self.org_data.values[:,index:]
        label = self.org_data.values[:,index-1:index] if self.role == 'guest' else None
        return feature, label

    @property
    def shape(self):
        return self.feature.shape