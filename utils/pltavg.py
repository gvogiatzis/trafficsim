#!/usr/bin/env python

import typer
from typing import Optional as Opt, List, Tuple, Union
from typing_extensions import Annotated as Ann
import sys

app = typer.Typer(context_settings={"help_option_names": ["-h", "--help"]}, add_completion=False)

@app.command()
def main(fnames: Ann[List[str], typer.Argument(help="the filename of the csv file whose columns will be averaged and plotted. If omitted will read from stdin.")]=None,
         x_label: Ann[str, typer.Option("--xlabel", "-x", help="Label of the x axis. Defaults to 'time' if not supplied")] = "time",
         y_label: Ann[str, typer.Option("--ylabel", "-y", help="Label of the y axis. Defaults to 'value' if not supplied")] = "value",
         title: Ann[str, typer.Option("--title", "-t", help="Title of the plot. Defaults to '' if not supplied")] = "",
         separator: Ann[str, typer.Option("--sep", help="Separator character for row fields.")] = ",",
         smoothing: Ann[int, typer.Option("--smoothing", "-s", help="If set, will compute a rolling sum with a sliding window of size equal to the parameter.")] = None,
         has_header: Ann[bool, typer.Option(help="If set, assumes the first row of the table has the header names.")] = False,
         index_col:Ann[str, typer.Option("--index", "-i", help="If set, provides the name of the column that will be used as index to the data. The table must have valid headers and the collumn header name must be used. If not set a default 0..N integer index will be used.")] = None):
    """
    A simple script for plotting the average of a set of time series. These must be encoded as a table with T rows and N columns, where N is the number of different time series and T is the number of elements in each of them. Can read the input from a csv file or from stdin. 
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    fnames = fnames if fnames else [sys.stdin]
    for fname in fnames:
        header = 0 if has_header else None
        # df = pd.read_csv(fname, sep=",")
        df = pd.read_csv(fname, sep=separator, header=header, index_col=index_col)
        df.index.name = x_label
        if smoothing:
            df = df.rolling(smoothing).mean()
        df = df.stack().reset_index(level=1)
        df = df.rename(columns={0:y_label})
        # print(df.columns)
        df = df.drop(columns=['level_1'])
        # print(df)
        # return
        if len(fnames)>1:
            _ = sns.lineplot(data=df, x=x_label, y=y_label,label=fname)
        else:
            _ = sns.lineplot(data=df, x=x_label, y=y_label)

    # print(df, file=sys.stderr)
    # plt.show()
    plt.title(title)
    plt.savefig(sys.stdout.buffer)

if __name__=="__main__":
    app()