from datetime import datetime
from multiprocessing import Pool
import time

lst = [(2, 2),  (4, 4), (5, 5),(6,6),(3, 3),]
result = []

def mul(x):
    print(f"start process {x}")
    time.sleep(3)
    print(f"end process {x}")
    res = x[0] * x[1]
    res_ap = (x[0] , x[1] , res)
    return res_ap

# Map
def test_map():
    pool = Pool(processes=10)
    res = pool.map(mul, lst,chunksize=2)
    result.append(res)
    pool.close()
    pool.join()
    print(result)

if __name__ == '__main__':
    start = datetime.now()
    test_map()
    print("End Time Map:", (datetime.now() - start).total_seconds())