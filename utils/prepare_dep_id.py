
import json
import helper

data_dir = 'dataset/tacred'


def load_deprel(filename, data_dir,deprel):
    with open(filename) as infile:
        data = json.load(infile)
        for d in data:
            deps = d['stanford_deprel']
            for dep in deps:
                if dep not in deprel:
                    deprel.append(dep)
        print("{} tokens from {} examples loaded from {}.".format(len(deprel), len(data), filename))
    return deprel

if __name__ == '__main__':

    deprel = []
    # input files
    train_file = data_dir + '/train.json'
    dev_file = data_dir + '/dev.json'
    test_file = data_dir + '/test.json'



    # load files
    print("loading files...")
    train_tokens = load_deprel(train_file, data_dir,deprel)
    dev_tokens = load_deprel(dev_file, data_dir,deprel)
    test_tokens = load_deprel(test_file, data_dir,deprel)
    print(deprel)

    deprel2id = {"self":1}
    count=2
    for dep in deprel:
        deprel2id[dep] = count

        count += 1
    print(deprel2id)





