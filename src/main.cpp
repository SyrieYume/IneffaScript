#include <memory_resource>
#include <print>
#include <fstream>
#include <ranges>
#include <string_view>
#include "vm.hpp"
#include "parser.hpp"
#include "win32_x64_jit.hpp"


auto read_file(const std::string_view file_path) -> std::string {
    std::string result;

    std::ifstream file(std::string(file_path), std::ios::binary | std::ios::ate);
    
    if(!file.is_open())
        throw std::runtime_error(std::format("faild to open '{}'", file_path));

    size_t file_size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    result.resize(file_size);

    if(!file.read(result.data(), file_size))
        throw std::runtime_error(std::format("faild to read '{}'", file_path));

    return result;
}


// 反汇编字节码
static auto disassembly(const ineffa::script::executable_t& exe) -> std::string {
    using namespace ineffa::script;
    using enum opcode_t;

    std::string result;
    result.reserve(128);

    const auto& [bytecodes, constants, functions] = exe;
    
    for (auto [i, ins] : bytecodes | std::views::enumerate) {
        std::format_to(std::back_inserter(result), "{:04d}: {:<28}", i, opcode_to_string_array[(uint8_t)ins.opcode]);

        switch (ins.opcode) {
            case load_64: 
            case load_u32:
            case load_u16:
            case load_u8: 
            case load_i32:
            case load_i16:
            case load_i8: 

            case store_64:
            case store_32:
            case store_16:
            case store_8:
            case add_imm_i8:
                std::format_to(std::back_inserter(result), "r{}, r{}, {}\n", ins.rd, ins.i.rs1, ins.i.imm);
                break;

            case load_imm_i16:
                std::format_to(std::back_inserter(result), "r{}, {}\n", ins.rd, ins.u.imm);
                break;

            case load_global_64:
            case store_global_64:
                std::format_to(std::back_inserter(result), "r{}, {}\n", ins.rd, (uint16_t)ins.u.imm);
                break;
            
            case load_constant_str:
                std::format_to(std::back_inserter(result), "r{}, data: {:?}\n", ins.rd, (const char*)&constants[(uint16_t)ins.u.imm * 8]);
                break;

            case load_constant_64:
                std::format_to(std::back_inserter(result), "r{}, data: {}\n", ins.rd, *(uint64_t*)&constants[(uint16_t)ins.u.imm * 8]);
                break;
            
            case not_64:
            case move_64:
            case cast_i64_to_f64:
            case cast_f64_to_i64:
                std::format_to(std::back_inserter(result), "r{}, r{}\n", ins.rd, ins.i.rs1);
                break;

            case jump:
                std::format_to(std::back_inserter(result), "target: {:04d}\n", (int32_t)i + (int32_t)ins.u.imm + 1);
                break;

            case call:
                std::format_to(std::back_inserter(result), "target: {:04d}, bp: r{}\n", (int32_t)i + (int32_t)ins.u.imm + 1, ins.rd);
                break;

            case opcode_t::call_reg:
                std::format_to(std::back_inserter(result), "target: r{}, bp: r{}\n", ins.rd, (uint16_t)ins.u.imm);
                break;

            case opcode_t::jump_if_true: 
            case opcode_t::jump_if_false:
                std::format_to(std::back_inserter(result), "r{}, target: {:04d}\n", ins.rd, (int32_t)i + (int32_t)ins.u.imm + 1);
                break;
                
            case jump_if_greater_than_i64:
            case jump_if_greater_equal_i64:
                std::format_to(std::back_inserter(result), "r{}, r{}, target: {:04d}\n", ins.rd, ins.i.rs1, (int32_t)i + (int32_t)ins.i.imm + 1);
                break;

            case loop_inc_check_jump:
                std::format_to(std::back_inserter(result), "r{}, r{}, target: {:04d}\n", ins.rd, ins.i.rs1, (int32_t)i - (uint8_t)ins.i.imm + 1);
                break;
            
            case opcode_t::call_host:
                std::format_to(std::back_inserter(result), "func_id: {}, args: r{}\n", (uint16_t)ins.u.imm, ins.rd);
                break;

            case opcode_t::ret:
            case opcode_t::halt:
                std::format_to(std::back_inserter(result), "\n");
                break;

            default:
                std::format_to(std::back_inserter(result), "r{}, r{}, r{}\n", ins.rd, ins.r.rs1, ins.r.rs2);
                break;
        }
    }
    return result;
}


auto main(int argc, char* argv[]) -> int try {
    using namespace ineffa::script;

    std::vector<std::string_view> args(argv + 1, argv + argc);

    if (args.empty())
        throw std::runtime_error("no input file specified");

    std::string_view mode = "--run-jit";
    std::string_view file_path;

    if (args[0].starts_with("--")) {
        mode = args[0];
        if (args.size() < 2)
            throw std::runtime_error(std::format("missing filename after '{}'", mode));
        file_path = args[1];
    }
    else {
        file_path = args[0];
    }

    vm_t vm = vm_t(8 * 1024);

    vm.host_functions.emplace_back("get_time", [](value_t* args, value_t* result) -> void {
        *result = { .f64 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count() };
    });

    vm.host_functions.emplace_back("print_str", [](value_t* args, value_t* result) -> void {
        std::print("{}", (const char*)args[0].u64); 
    });

    vm.host_functions.emplace_back("print_i64", [](value_t* args, value_t* result) -> void {
        std::print("{}", args[0].i64);
    });

    vm.host_functions.emplace_back("print_f64", [](value_t* args, value_t* result) -> void {
        std::print("{:.6f}", args[0].f64);
    });

    
    std::string source_code = read_file(file_path);

    std::pmr::monotonic_buffer_resource memory_pool = std::pmr::monotonic_buffer_resource(16 * 1024, std::pmr::new_delete_resource());
    std::pmr::polymorphic_allocator<std::byte> memory_allocator(&memory_pool);
    
    try {
        ast_node_t* root_node = ast_parser_t(memory_allocator).parse(source_code);
        executable_t exe = parser_t().parse(root_node, vm.host_functions);

        if (mode == "--assembly")
            std::print("{}", disassembly(exe));
            
        else if (mode == "--run")
            vm.run(exe);

        else if (mode == "--run-jit") {
            auto jit_func = win32_x64_jit::compile(exe, vm.host_functions);
            jit_func(vm.bp, vm.sp);
        }

        else throw std::runtime_error(std::format("unknown argument '{}'", mode));
    }

    catch(ineffa::script::parser_t::error_t& e) {
        std::string_view atom = e.error_node->is_atom() ? e.error_node->get_atom() : e.error_node->get_list()[0]->get_atom();
        std::string_view prefix(source_code.data(), atom.data());
        int line = std::count(prefix.begin(), prefix.end(), '\n') + 1;
        int col = prefix.size() - prefix.rfind('\n'); 
        std::println("[line = {}, col = {}, atom = {}] {}", line, col, atom, e.what());
    }

    return 0;
}

catch(std::exception& e) {
    std::println("error: {}", e.what());
    return -1;
}