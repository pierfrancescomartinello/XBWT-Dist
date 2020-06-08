import argparse
import pandas as pd
import numpy as np


def plot_descend(df, axis_labels, titles, outfile=None):
    import seaborn as sns
    import matplotlib.pyplot as plt

    ax = sns.lineplot(data=df, x='Tree', y='Similarity', style='Measure', hue='Measure',
                      markers=True,
                      palette="Set2")

    ax.set(xticks=range(8), xticklabels=[
           'T$_0$', 'T$_1$', 'T$_2$', 'T$_3$', 'T$_4$', 'T$_5$', 'T$_6$', 'T$_7$'])

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()


parser = argparse.ArgumentParser(description='plot figures', add_help=True)
parser.add_argument('--csvs', action='store', nargs='+', required=True,
                    help='path of the csv files')
parser.add_argument('--names', action='store', nargs='+', required=True,
                    help='names of the measures')
parser.add_argument('-o', '--outfile', action='store', type=str, default=None,
                    help='output file path.')
args = parser.parse_args()

assert(
    len(args.csvs) == len(args.names)
)

df = pd.DataFrame(columns=['Tree', 'Similarity', 'Measure'])

lines = list()
for ix, csv in enumerate(args.csvs):
    with open(csv) as fin:
        x = [float(x.strip()) for x in fin.readlines()]
    lines.append(x)
    for xx, v in enumerate(x):
        df.loc[len(df)] = [xx, v, args.names[ix]]
        # print(f'{xx},{v},{args.names[ix]}')

# print(df)
plot_descend(df, None, args.names, outfile=args.outfile)
