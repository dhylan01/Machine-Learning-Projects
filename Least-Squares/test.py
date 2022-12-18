
if __name__ == '__main__':
    ret = {"S": 10,
           "N": 5,
           "O": 23,
           "W": 3, "E": 100, "F": 50, "A": 25, "Z": 2, "T": 6}

    x = set()
    p = ""
    for char in ret.keys():
        p += char * ret[char]
    print(p)
