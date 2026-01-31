import { performance } from 'perf_hooks';

function isPrime(n) {
    let i = 2;
    while (true) {
        if (i * i > n)
            break
        if (n % i === 0)
            return false;
        i++;
    }
    return true;
}

const startTime = performance.now();
let totalCount = 0;

for (let currentNum = 2; currentNum < 10000000; currentNum++) {
    if (isPrime(currentNum))
        totalCount++;
}

const endTime = performance.now();
const elapsedMs = endTime - startTime;

console.log(`found primes count: ${totalCount}`);
console.log(`time used: ${elapsedMs.toFixed(6)}ms`);
