#pragma once
#include <vector>
#include <cassert>
#include "enum_magic.hpp"


namespace ineffa::script {

generate_enum_class_and_map(uint8_t, opcode_t, opcode_to_string_array, string_to_opcode_map, 
    add_64,
    sub_64,
    mul_64,
    div_u64,
    div_i64,

    and_64,
    or_64,
    xor_64,
    not_64,

    shl_64,
    shr_u64,
    shr_i64,

    mod_u64,
    mod_i64,

    add_f64,
    sub_f64, 
    mul_f64, 
    div_f64,

    add_imm_i8,
    load_imm_i16,

    move_64,

    set_is_less_than_u64,
    set_is_less_than_i64,
    set_is_less_than_f64,
    set_is_less_equal_u64,
    set_is_less_equal_i64,
    set_is_less_equal_f64,
    set_is_equal_64,
    set_is_not_equal_64,

    jump,
    jump_if_true,
    jump_if_false,
    jump_if_greater_than_i64,
    jump_if_greater_equal_i64,
    loop_inc_check_jump,

    load_64,
    load_u32,
    load_i32,
    load_u16,
    load_i16,
    load_u8,
    load_i8,

    store_64,
    store_32,
    store_16,
    store_8,
    
    load_global_64,
    store_global_64,
    load_constant_str,
    load_constant_64,


    cast_i64_to_f64,
    cast_f64_to_i64,

    call,
    call_reg,
    call_host,
    ret,

    halt,
);

struct instruction_t {
    using ins_type_r_t = struct {
        uint8_t rs1;     // 源虚拟寄存器1：数据1相对当前栈帧的偏移量
        uint8_t rs2;     // 源虚拟寄存器2：数据2相对当前栈帧的偏移量
    };

    using ins_type_i_t = struct {
        uint8_t rs1;    // 基址 / 条件比较源虚拟寄存器2
        int8_t imm;     // 小的立即数
    };

    using ins_type_u_t = struct {
        int16_t imm;    // 较大的立即数
    };

    opcode_t opcode; // 指令
    uint8_t rd;      // 目标虚拟寄存器 / 条件比较源虚拟寄存器1
    union {
        ins_type_r_t r;
        ins_type_i_t i;
        ins_type_u_t u;
    };

    instruction_t(opcode_t opcode, uint8_t rd, uint8_t rs1, uint8_t rs2) : opcode(opcode), rd(rd), r(rs1, rs2) {}
    instruction_t(opcode_t opcode, uint8_t rd, uint8_t rs1, int8_t imm) : opcode(opcode), rd(rd), i(rs1, imm) {}
    instruction_t(opcode_t opcode, uint8_t rd, int16_t imm) : opcode(opcode), rd(rd), u(imm) {}
};

static_assert(std::endian::native == std::endian::little && sizeof(instruction_t) == 4);

struct value_t {
    union {
        const void* ptr;
        uint64_t u64;
        int64_t i64;
        double f64;

        uint32_t u32;
        int32_t i32;
        float f32;

        uint16_t u16;
        int16_t i16;
        uint8_t u8;
        int8_t i8;
    };
};

struct function_info_t {
    uint32_t start, end;
    uint32_t args_size;
    uint32_t ret_val_size;
};

struct executable_t {
    std::vector<instruction_t> bytecodes;
    std::vector<uint8_t> constants;
    std::vector<function_info_t> functions;
};

using host_func_t = void(*)(value_t* args, value_t* result);
}