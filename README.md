# IneffaScript
一个简单的脚本语言实现

## 使用方法
```powershell
# JIT执行
IneffaScript.exe 源代码.lsp
IneffaScript.exe --run-jit 源代码.lsp

# 解释执行
IneffaScript.exe --run 源代码.lsp

# 打印编译的字节码
IneffaScript.exe --assembly 源代码.lsp
```

## 语法样例
### 统计素数个数（`test/count_primes.lsp`）
```
(defun is-prime (n :i64) :bool
  (let i :i64 = 2)
  (loop 
    (if ((i * i) > n)
      break
    )

    (if ((n % i) == 0) 
      (return false)
    )
    
    (i = i + 1)
  )

  (return true)
)

(let start-time :f64 = (callhost get_time))
(let total-count :i64 = 0)

(for let current-num :i64 = 2 to 1000000
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
```

### 冒泡排序（`test/array_sort.lsp`）
```
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
```

## 平台支持
JIT的部分目前只支持 **Windows x86_64**，理论上只需要很小的修改就可以移植到 **Linux x86_64** (暂未测试)

删除JIT的部分理论上可以支持所有64位小端CPU平台

## 如何编译本项目
1. 需要支持 **C++23标准** 的 **C++编译器**

2. Clone 本项目: 
    ```powershell
    git clone --depth 1 https://github.com/SyrieYume/IneffaScript.git
    cd IneffaScript
    ```

3. 执行下面的编译指令：
    ```powershell
    # 如果编译器是 clang
    clang++ -std=c++23 -O3 src/main.cpp -o IneffaScript.exe

    # 如果编译器是 gcc
    g++ -std=c++23 -O3 src/main.cpp -o IneffaScript.exe -lstdc++exp --static
    ```
