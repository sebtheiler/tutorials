import math
import multiprocessing
import time

import numpy as np
import SharedArray

x_dim, y_dim = 100, 1000000

fake_data = np.random.random((x_dim, y_dim))


def process_data(fake_data):
    data = fake_data.copy()
    for i, row in enumerate(data):
        for j, data_point in enumerate(row):
            if data_point == None:
                data[i, j] = 0
            data[i, j] = math.exp(math.sqrt(data_point))
    return data


def multiprocess_data(fake_data):
    data = SharedArray.create('data', (x_dim, y_dim))

    def calc_row(i):
        for j, data_point in enumerate(fake_data[i]):
            if data_point == None:
                data[i, j] = 0
            data[i, j] = math.exp(math.sqrt(data_point))
    
    processes = []
    for i in range(len(fake_data)):
        process = multiprocessing.Process(target=calc_row, args=(i,))
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    
    return data


if __name__ == "__main__":
    start = time.time()
    process_data(fake_data)
    end = time.time()

    print(end - start)


    start = time.time()
    multiprocess_data(fake_data)
    end = time.time()

    print(end - start)

    SharedArray.delete('data')
