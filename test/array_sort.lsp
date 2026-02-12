(let size :i32 = 12)
(let arr :i32[] = callhost new (size * 4))

(for let i :i32 = 0 to size
    ((arr at i) = callhost random_i32 -100 100)
)

(callhost print_str "unsorted array: \n")
(for let i :i32 = 0 to size
    (callhost print_i64 (arr at i))
    (callhost print_str " ")
)
(callhost print_str "\n")

(for let i :i32 = 0 to size
    (for let j :i32 = 0 to ((size - 1) - i)
        (let current :i32 = arr at j)
        (let next :i32 = arr at (j + 1))

        (if (current > next)
            ((arr at j) = next)
            ((arr at (j + 1)) = current)
        )
    )
)

(callhost print_str "sorted array: \n")
(for let i :i32 = 0 to size
    (callhost print_i64 (arr at i))
    (callhost print_str " ")
)
(callhost print_str "\n")

(callhost delete arr)