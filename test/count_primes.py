import time

def is_prime(n):
    i = 2
    while True:
        if i * i > n:
            break
        if n % i == 0:
            return False
        i += 1
        
    return True

start_time = time.time()

total_count = 0
current_num = 2

for current_num in range(2, 10000000):
    if is_prime(current_num):
        total_count += 1
    current_num += 1

end_time = time.time()

elapsed_ms = (end_time - start_time) * 1000

print(f"found primes count: {total_count}")
print(f"time used: {elapsed_ms:.6f}ms")