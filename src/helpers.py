import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


def find_columns_type(df: pd.DataFrame) -> Tuple[List, List, List, List]:
    numerical_columns = []
    categorical_columns = []
    bool_columns = []
    other_types_columns = []

    for i in df.columns:
        if str(df[i].dtype) == 'float64' or str(df[i].dtype) == 'int64':
            numerical_columns.append(i)
        elif str(df[i].dtype) == 'object':
            categorical_columns.append(i)
        elif str(df[i].dtype) == 'bool':
            bool_columns.append(i)
        else:
            other_types_columns.append(i)

    return numerical_columns, categorical_columns, bool_columns, other_types_columns


def find_na_columns(df: pd.DataFrame) -> Dict:
    na_columns = {}

    for i in df.columns:
        if df[i].isna().sum() > 0:
            na_columns[i] = df[i].isna().sum()
        elif df[i][df[i] == 'None'].shape[0] > 0:
            na_columns[i] = df[i][df[i] == 'None'].shape[0]

    return na_columns


def distribution(df: pd.DataFrame, col: str) -> pd.DataFrame:
    total_clients = df['order_id'].nunique()
    grouped = df.groupby([col, 'typical'])['order_id'].nunique()
    total = df.groupby(col)['order_id'].nunique()
    result = (grouped / total).reset_index()
    result = result.rename(columns={'order_id': 'dist_in_group'})
    total = total.reset_index()
    result = pd.merge(result, total, on=col, how='left')
    result = result.rename(columns={'order_id': 'dist_in_total'})
    result['dist_in_total'] = result['dist_in_total'] / total_clients
    result = result.sort_values(by='dist_in_total', ascending=False)

    return result


def add_buckets(df, quantiles, score_column):
    df['bucket'] = np.nan

    def check_df_type(df_tmp):
        df_tmp['bucket'][(df_tmp[score_column] >= lower) & (df_tmp[score_column] < upper)] = idx
        return df_tmp

    bucket_range = []

    for idx, i in enumerate(quantiles):
        if idx == 0:
            lower = i
        else:
            upper = i
            df = check_df_type(df)
            bucket_range.append([idx, lower, upper])
            lower = upper

    return bucket_range, df


def check_psi_value(psi_value):
    if psi_value < .1:
        psi_result = 'Distribution is stable'
    elif psi_value < .2:
        psi_result = 'Changes in distribution'
    else:
        psi_result = 'Critical changes in distribution'

    return psi_result


def check_dpd_value(dpd_value):
    if dpd_value < .1:
        dpd_result = 'Distribution is stable'
    elif dpd_value < .25:
        dpd_result = 'Some changes in distribution'
    elif dpd_value < .5:
        dpd_result = 'Significant changes in distribution'
    else:
        dpd_result = 'Critical changes in distribution'

    return dpd_result
