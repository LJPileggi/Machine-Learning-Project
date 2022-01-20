from multiprocessing import Pool
import os
import signal
import sys

class KeyboardInterruptError(Exception): pass

def get_primes(m, i):
    #signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        primes = [2]
        print(f"process {i}: computation started")
        for n in range(3, m+1):
            is_prime = True
            for p in primes:
                if n % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(n)
        return primes
    except KeyboardInterrupt:
        print(f"process {i}: interrupted")
        return None


def initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    os._exit(1)



def main():
    signal.signal(signal.SIGINT, signal_handler)
    m = 200000
    # get_primes(m)
    pool = Pool()
    # try:
    results = [
        pool.apply_async(get_primes, (m, i))
        for i in range(8) ]
    pool.close()
    pool.join()
    print(results[1].get())
    print("done")
    # except KeyboardInterrupt:
    #     print("main process: detected interrupt")
    #     pool.terminate()
    #     #pool.join()
    #     print("main process: pool terminated")
    # finally:
    #     print('joining pool processes')
    #     pool.join()
    #     print('join complete')

if __name__ == "__main__":
    main()