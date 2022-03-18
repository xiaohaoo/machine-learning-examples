import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('datasets/Nafta.csv')

    with open('datasets/sequential_data.csv', 'w+') as f:
        f.write('v1,v2,v3,v4,v5,v6,v7\n')
        for i, k in enumerate(data['price'][:-7]):
            row = []
            for j in range(7):
                f.write(str(data['price'][i + j]))
                if j != 6:
                    f.write(',')
            f.write('\n')
