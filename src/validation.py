import numpy as np
import pandas as pd
from .helpers import add_buckets, check_psi_value, check_dpd_value


class DataValidation:
    def __init__(self, control, test, score_column='score', num_bins=10, metric='PSI'):
        self.control = control
        self.test = test
        self.score_column = score_column
        self.num_bins = num_bins
        self.metric = metric

    def calculate_stability(self):
        quantiles = [self.control[self.score_column].quantile(i/self.num_bins) for i in range(self.num_bins + 1)]
        quantiles[0] = float('-inf')
        quantiles[-1] = float('inf')

        bucket_range, control_with_buckets = add_buckets(self.control, quantiles, self.score_column)
        control_total = control_with_buckets.count()[0]

        control_buckets = control_with_buckets.groupby('bucket').aggregate(
            control_freq=(self.score_column, 'count')).reset_index()

        bucket_range_df = pd.DataFrame(bucket_range, columns=['bucket', 'bucket_low', 'bucket_high'])
        control_buckets = pd.merge(bucket_range_df, control_buckets, on='bucket', how='left')
        control_buckets['control_dist'] = control_buckets['control_freq'] / control_total

        _, test_with_buckets = add_buckets(self.test, quantiles, self.score_column)
        test_total = test_with_buckets.count()[0]

        test_buckets = test_with_buckets.groupby('bucket').aggregate(
            test_freq=(self.score_column, 'count')).reset_index()

        test_buckets['test_dist'] = test_buckets['test_freq'] / test_total

        buckets = pd.merge(control_buckets, test_buckets, on='bucket', how='left')

        if self.metric == 'PSI':
            buckets['diff'] = buckets['test_dist'] - buckets['control_dist']
            buckets['ln(A/B)'] = np.log(buckets['test_dist'] / buckets['control_dist'])
            buckets['CSI'] = buckets['diff'] * buckets['ln(A/B)']
            metric_value = buckets['CSI'].sum()
            result = check_psi_value(metric_value)
        elif self.metric == 'DPD':
            buckets['DPD'] = ((buckets['test_dist'] - buckets['control_dist']) / np.sqrt(buckets['control_dist'])) ** 2
            metric_value = buckets['DPD'].sum()
            result = check_dpd_value(metric_value)
        else:
            metric_value = np.nan
            result = 'Metric is not supported'

        return result, metric_value, buckets