local function is_prime(n)
    local i = 2
    while true do
        if (i * i) > n then
            break
        end

        if n % i == 0 then
            return false
        end
        i = i + 1
    end

    return true
end

local start_time = os.clock()
local total_count = 0

for current_num = 2, 10000000 do
    if is_prime(current_num) then
        total_count = total_count + 1
    end
end

local end_time = os.clock()
local elapsed_ms = (end_time - start_time) * 1000

print(string.format("found primes count: %d", total_count))
print(string.format("time used: %.6fms", elapsed_ms))