(defun is-prime (n :i64) :bool
  (let i :i64 = 2)
  (loop 
    (if ((i * i) > n)
      break
    )

    (if ((n % i) == 0) 
      (return false))
    
    (i = i + 1)
  )

  (return true)
)

(let start-time :f64 = (callhost get_time))
(let total-count :i64 = 0)

(for let current-num :i64 = 2 to 10000000
  (if (call is-prime current-num)
    (total-count = total-count + 1)
  )
)

(let end-time :f64 = (callhost get_time))
(callhost print_str "found primes count: ")
(callhost print_i64 total-count)
(callhost print_str "\n")
(callhost print_str "time used: ")
(callhost print_f64 (end-time - start-time))
(callhost print_str "ms")