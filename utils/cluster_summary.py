import pandas as pd

def get_cluster_summary(df):

    summary = pd.pivot(
        data=df.groupby(["label", "pred"], as_index=False).size(),
        index="pred",
        columns="label",
    ).fillna(0)
    return summary.div(summary.values.sum(axis=0), axis=1)