# NOTE: use --configfile to specify the yaml of configs
# NOTE: run with --restart-times 5 because similations will fail

import re
import numpy as np
import networkx as nx

data_path = config["dataout"]
simulate_params = config['simulate_params']


mp3 = ['mp3']
stereodist_tool = ['CASet_I', 'DISC_I', 'CASet_U', 'DISC_U']
mlted = ['MLTED']
tot_pb = int(config['tot_pb'])
tot_pb_trees = range(1, int(config['tot_pb']) + 1)

base_trees = ['base_tree1', 'base_tree2',
              'base_tree3', 'base_tree4', 'base_tree5']


def build_calc_files(tool_name):
    files_lst = list()
    for bi in range(1, len(base_trees) + 1):
        for bj in range(bi, len(base_trees) + 1):
            for i in range(1, tot_pb + 1):
                x = i if bi == bj else 1
                for j in range(x, tot_pb + 1):
                    files_lst.append(
                        data_path +
                        f'/{tool_name}/base_tree{bi}.base_tree{bj}.p{i}.p{j}.out'
                    )

    return files_lst


# rule plot_all:
#     output: data_path + '/heatmaps.pdf'
#     input:
#         expand(data_path + '/{tool}/results.csv',
#                tool=mp3 + stereodist_tool + mlted)
#     params:
#         names = ' '.join(mp3 + stereodist_tool + mlted)
#     shell:
#         """
#         python3 plot_heatmaps.py \
#             --csv {input} --names {params.names} \
#             -o {output}
#         """

rule run_all:
    input:
        expand(data_path + '/{tool}/results.csv',
               tool=mp3 + stereodist_tool + mlted)

#   MLTED
#

rule build_csv_mlted:
    output:
        csv = data_path + '/MLTED/results.csv'
    input:
        files = build_calc_files('MLTED')
    run:
        outfile = output.csv
        files = input.files
        dict_file = dict()
        base_sets = set()
        pb_sets = set()

        search = re.compile(
            r'base_tree(?P<base1>\d+).base_tree(?P<base2>\d+).p(?P<pb1>\d+).p(?P<pb2>\d+).out')
        for file in files:
            m = search.search(file)
            with open(file, 'r') as fin:
                lines = fin.readlines()
                if len(lines) > 0:
                    value = float(lines[-2].strip().split()[-1])
                else:
                    value = -1
                dict_file[(int(m.group('base1')) - 1, int(m.group('base2')) - 1,
                           int(m.group('pb1')) - 1, int(m.group('pb2')) - 1)] = value

                base_sets.add(int(m.group('base1')) - 1)
                base_sets.add(int(m.group('base2')) - 1)
                pb_sets.add(int(m.group('pb1')) - 1)
                pb_sets.add(int(m.group('pb2')) - 1)

        base_sets = sorted(list(base_sets))
        pb_sets = sorted(list(pb_sets))

        matrix = np.zeros((len(base_sets) * len(pb_sets),
                           len(base_sets) * len(pb_sets)))

        for k in dict_file:
            b1, b2, pb1, pb2 = k
            i = b1 * len(pb_sets) + pb1
            j = b2 * len(pb_sets) + pb2

            matrix[i, j] = dict_file[k]
            matrix[j, i] = dict_file[k]

        np.savetxt(outfile, matrix, delimiter=',')


rule run_mlted:
    output: temp(data_path + '/MLTED/{bt_i}.{bt_j}.p{i}.p{j}.out')
    input:
        tree1 = data_path + '/MLTED/{bt_i}/{bt_i}.p{i}.txt',
        tree2 = data_path + '/MLTED/{bt_j}/{bt_j}.p{j}.txt',
    shell:
        """
        ./measures/MLTED/MLTED {input.tree1} {input.tree2} > {output} || true  2> /dev/null
        """

rule convert_to_mlted:
    output:
        txt = temp(data_path + '/MLTED/{bt_i}/{bt_i}.p{i}.txt')
    input:
        gv = data_path + '/{bt_i}/{bt_i}.p{i}.gv'
    run:
        t_path = input.gv
        T = nx.DiGraph(nx.drawing.nx_agraph.read_dot(t_path))
        with open(output.txt, 'w+') as fout:
            for node in T.nodes:
                label = T.nodes[node]['label']
                print(f'{node}={label}', file=fout)

            for line in nx.generate_adjlist(T, delimiter=' '):
                x = line.split()
                if len(x) > 1:
                    node = f'{x[0]}'
                    adjlist = ','.join(x[1:])
                    print(f'{node}:{adjlist}', file=fout)


#
#   StereoDist
#

rule run_stereodist:
    output:
        csv = data_path + '/{tool}/results.csv'
    input: data_path + '/{tool}/trees.nw'
    params:
        intermediary = data_path + '/{tool}/results.tsv',
        tool = '{tool}'
    run:
        u = '-u' if 'U' in params.tool else ''
        tool_path = params.tool[:-2]
        shell(
            'python3 measures/stereodist/{tool_path}.py -o {params.intermediary} {input} {u}'
        )

        lines = list()
        with open(params.intermediary) as fin:
            lines = fin.readlines()

        # skip first 2
        lines = lines[2:]

        trees = len(lines[0].split('\t')) - 1
        matrix = np.zeros((trees, trees))

        for row, line in enumerate(lines):
            for col, value in enumerate(line.strip().split('\t')[1:]):
                matrix[row][col] = 1 - float(value)

        outfile = output.csv
        np.savetxt(outfile, matrix, delimiter=',')

rule convert_to_stereodist:
    output:
        converted = data_path + '/{tool}/trees.nw'
    input:
        files = expand(data_path + '/{bt_i}/{bt_i}.p{i}.gv',
                       bt_i=base_trees,
                       i=tot_pb_trees)
    run:
        def get_label(g, node):
            label = g.nodes[node]['label']
            if ',' in label:
                return '{' + label + '}'
            else:
                return label

        def tree_to_newick(g, root=None):
            if root is None:
                roots = list(filter(lambda p: p[1] == 0, g.in_degree()))
                assert 1 == len(roots)
                root = roots[0][0]
            subgs = []
            for child in g[root]:
                if len(g[child]) > 0:
                    subgs.append(tree_to_newick(g, root=child))
                else:
                    subgs.append(get_label(g, child))
            return "(" + ','.join(subgs) + ")" + get_label(g, root)

        outfile = output.converted
        with open(outfile, 'w+') as fout:
            for tree in input.files:
                T = nx.DiGraph(nx.drawing.nx_agraph.read_dot(tree))
                fout.write(tree_to_newick(T, None) + ';\n')

#
#   MP3
#

rule build_csv_mp3:
    output:
        csv = data_path + '/mp3/results.csv'
    input:
        files = build_calc_files('mp3')
    run:
        outfile = output.csv
        files = input.files
        dict_file = dict()
        base_sets = set()
        pb_sets = set()

        search = re.compile(
            r'base_tree(?P<base1>\d+).base_tree(?P<base2>\d+).p(?P<pb1>\d+).p(?P<pb2>\d+).out')
        for file in files:
            m = search.search(file)
            with open(file, 'r') as fin:
                value = float(fin.readlines()[0].strip())
                dict_file[(int(m.group('base1')) - 1, int(m.group('base2')) - 1,
                           int(m.group('pb1')) - 1, int(m.group('pb2')) - 1)] = value

                base_sets.add(int(m.group('base1')) - 1)
                base_sets.add(int(m.group('base2')) - 1)
                pb_sets.add(int(m.group('pb1')) - 1)
                pb_sets.add(int(m.group('pb2')) - 1)

        base_sets = sorted(list(base_sets))
        pb_sets = sorted(list(pb_sets))

        matrix = np.zeros((len(base_sets) * len(pb_sets),
                           len(base_sets) * len(pb_sets)))

        for k in dict_file:
            b1, b2, pb1, pb2 = k
            i = b1 * len(pb_sets) + pb1
            j = b2 * len(pb_sets) + pb2

            matrix[i, j] = dict_file[k]
            matrix[j, i] = dict_file[k]

        np.savetxt(outfile, matrix, delimiter=',')

rule run_mp3:
    output: temp(data_path + '/mp3/{bt_i}.{bt_j}.p{i}.p{j}.out')
    input:
        tree1 = data_path + '/{bt_i}/{bt_i}.p{i}.gv',
        tree2 = data_path + '/{bt_j}/{bt_j}.p{j}.gv',
    shell:
        """
        mp3treesim {input.tree1} {input.tree2} > {output}
        """
