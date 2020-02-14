data_path = "simulated/duplication"

mp3 = ['mp3']
stereodist_tool = ['CASet_I', 'DISC_I', 'CASet_U', 'DISC_U']

tot_pb = 15
tot_pb_trees = range(1, tot_pb + 1)

base_trees = ['base_tree']


tot_pb_params = 5

rule run_all:
    input:
        expand(data_path + '/{tool}/results.csv',
               tool=mp3 + stereodist_tool
               )

#
#   StereoDist
#
rule run_stereoall:
    output: data_path + '/{tool}/results.csv',
    input:
        expand(data_path + '/{tool}/{base_tree}_pbp{pb_p}.results.csv',
               base_tree=base_trees,
               pb_p=range(1, tot_pb_params + 1),
               tool='{tool}'
               )
    run:
        import re
        search = re.compile(
            r'base_tree_pbp(?P<pb>\d+).results.csv')

        with open(str(output), 'w+') as fout:
            fout.write('Base,pb,Similarity\n')
            for f in input:
                m = search.search(f)
                base = m.group('pb')
                with open(f, 'r') as fin:
                    for ix, line in enumerate(fin):
                        fout.write(f'{base},{ix + 1},{line}')


rule run_stereodist:
    output:
        csv = data_path + '/{tool}/{base_tree}_pbp{pb_p}.results.csv',
        tsv = temp(data_path + '/{tool}/{base_tree}_pbp{pb_p}.results.tsv')
    input: data_path + '/{tool}/{base_tree}_pbp{pb_p}.trees.nw'
    params:
        tool = '{tool}'
    run:
        u = '-u' if 'U' in params.tool else ''
        tool_path = params.tool[:-2]
        tot_tail = tot_pb
        shell(
            """
            python3 measures/stereodist/{tool_path}.py -o {output.tsv} {input} {u} && \
            cat {output.tsv} |  tr '\t' ' ' | cut -f2 -d' ' | tail -{tot_tail} | while read line ; do awk -vline=$line \'BEGIN{{printf "%.5f\\n", (1 - line)}}\' ; done > {output.csv}
            """
        )

#
#   TreeSim
#


rule mp3_all:
    output: data_path + '/mp3/results.csv'
    input:
        files = expand(data_path + '/mp3/{base_tree}_pbp{pb_p}.out',
                       base_tree=base_trees,
                       pb_p=range(1, tot_pb_params + 1))
    run:
        with open(str(output), 'w+') as fout:
            fout.write('Base,pb,Similarity\n')
            for f in input.files:
                with open(f, 'r') as fin:
                    fout.writelines(fin.readlines())

rule run_mp3:
    output: temp(data_path + '/mp3/{base_tree}_pbp{pb_p}.out')
    input:
        base = data_path + '/{base_tree}.gv',
    run:
        tot = tot_pb
        pb_p = wildcards.pb_p
        base_tree = wildcards.base_tree
        path = data_path
        tree = data_path + \
            '/{base_tree}_pbp{pb_p}/{base_tree}_pbp{pb_p}.p{i}.gv'
        for i in range(1, tot_pb + 1):
            shell(
                """
                echo -n '{pb_p},{i},' >> {output}
                mp3treesim {input.base} '{path}/{base_tree}_pbp{pb_p}/{base_tree}_pbp{pb_p}.p{i}.gv' >> {output}
                """
            )
