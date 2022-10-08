
import json
import helper

data_dir = '../dataset/cpr'


def count_len(filename, data_dir,lens):
    with open(filename) as infile:
        data = json.load(infile)
        for d in data:
            if len(d['token'])<6:
                print(d["token"])
            lens.append(len(d['token']))


if __name__ == '__main__':

    lens = []
    # input files
    train_file = data_dir + '/train.json'
    # dev_file = data_dir + '/dev.json'
    # test_file = data_dir + '/test.json'


    # load files
    print("loading files...")
    count_len(train_file, data_dir, lens)

    sub = 0
    minl = 1000
    maxl = 0
    for i in lens:
        sub += i
        minl = min(minl,i)
        maxl = max(maxl,i)
    midlen = sub/len(lens)
    print(minl)
    print(maxl)
    print(midlen)






