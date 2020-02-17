import sys, os, glob

def main():
    files = glob.glob(sys.argv[1])

    names = set()
    values = {}
    for f in files:
        run = os.path.dirname(f).split("/")[-1]
        tree1, tree2 = run.split("_")
        names.add(tree1)
        names.add(tree2)
        try:
            values[(tree1, tree2)] = round(float(open(f).readlines()[0].strip('\n')), 3)
        except IndexError:
            values[(tree1, tree2)] = float("nan")

    names = sorted(list(names))
    for n in names:
        print(',' + n, end='')
    print('')
    for i in range(0,len(names)):
        t1 = names[i]
        print(t1, end='')
        for j in range(0,len(names)):
            t2 = names[j]
            key = (t1, t2)
            if key not in values:
                key = (t2, t1)
            print(',' + str(values[key]), end='')
        print("")


if __name__ == "__main__":
    main()
