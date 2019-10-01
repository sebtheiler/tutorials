import multiprocessing
import os
import time

import cv2
import numpy as np
import SharedArray


def load_images():
    data = np.zeros((300, 1400, 2100, 3))
    files = os.listdir("train_images")
    for i, file in enumerate(files[:300]):
        image = cv2.imread("train_images/" + file)
        data[i] = image

    return data

def multiprocess_load_images():
    data = SharedArray.create('data', (300, 1400, 2100, 3))
    files = os.listdir("train_images")

    num_workers = 12
    worker_amount = int(300/num_workers)

    def load_images(i, n):
        to_load = files[i: i+n]
        for j, file in enumerate(to_load):
            data[i + j] = cv2.imread("train_images/" + file)
        
    processes = []
    for worker_num in range(num_workers):
        process = multiprocessing.Process(target=load_images, args=(worker_amount*worker_num, worker_amount))
        processes.append(process)
        process.start()
    
    for process in processes:
        process.join()
    
    return data

# Can't evaluate both during the same run, as most computers will run out of RAM
multiprocess = True

if __name__ == "__main__":
    start = time.time()

    if multiprocess == True:
        multiprocess_load_images()
    else:
        load_images()
    
    end = time.time()

    print(end - start)

    if multiprocess == True:
        SharedArray.delete('data')