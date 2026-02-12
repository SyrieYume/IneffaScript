#pragma once
#include <memory>
#include <string_view>
#include <utility>
#include "bytecode.hpp"

namespace ineffa::script {
class vm_t {
public:
    std::unique_ptr<value_t[]> value_stack;
    std::unique_ptr<uint64_t[]> call_stack;
    std::vector<std::pair<std::string_view, void(*)(value_t* args, value_t* result)>> host_functions;
    value_t* bp;
    uint64_t* sp;
    const instruction_t* pc;
    uint32_t stack_size;
    bool is_running;

    explicit vm_t(uint32_t stack_size) : 
        value_stack(new value_t[stack_size]),
        call_stack(new uint64_t[stack_size / 2]),
        bp(&value_stack[0]),
        sp(&call_stack[0]),
        pc(nullptr),
        stack_size(stack_size)
    {}

    void reset() {
        bp = &value_stack[0];
        sp = &call_stack[0];
        pc = nullptr;
    }

    __attribute__((aligned(64), noinline, hot)) void run(const executable_t& exe) {
        assert(!exe.bytecodes.empty() && exe.bytecodes.back().opcode == opcode_t::halt);

        this->is_running = true;

        using enum opcode_t;

        value_t* __restrict bp = std::assume_aligned<8>(this->bp);
        uint64_t* __restrict sp = std::assume_aligned<8>(this->sp);
        const instruction_t* __restrict pc = std::assume_aligned<4>(this->pc ? this->pc : exe.bytecodes.data());
        
        while (true) {
            const instruction_t ins = *(pc++);

            [[assume(ins.opcode <= opcode_t::halt)]];
            switch (ins.opcode) {
                case add_64: bp[ins.rd].u64 = bp[ins.r.rs1].u64 + bp[ins.r.rs2].u64; break;
                case sub_64: bp[ins.rd].u64 = bp[ins.r.rs1].u64 - bp[ins.r.rs2].u64; break;
                case mul_64: bp[ins.rd].u64 = bp[ins.r.rs1].u64 * bp[ins.r.rs2].u64; break;
                case div_u64: bp[ins.rd].u64 = bp[ins.r.rs1].u64 / bp[ins.r.rs2].u64; break;
                case div_i64: bp[ins.rd].i64 = bp[ins.r.rs1].i64 / bp[ins.r.rs2].i64; break;

                case and_64: bp[ins.rd].u64 = bp[ins.r.rs1].u64 & bp[ins.r.rs2].u64; break;
                case or_64:  bp[ins.rd].u64 = bp[ins.r.rs1].u64 | bp[ins.r.rs2].u64; break;
                case not_64: bp[ins.rd].u64 = ~bp[ins.r.rs1].u64; break;
                case xor_64: bp[ins.rd].u64 = bp[ins.r.rs1].u64 ^ bp[ins.r.rs2].u64; break;

                case shl_64:  bp[ins.rd].u64 = bp[ins.r.rs1].u64 << bp[ins.r.rs2].u64; break;
                case shr_u64: bp[ins.rd].u64 = bp[ins.r.rs1].u64 >> bp[ins.r.rs2].u64; break;
                case shr_i64: bp[ins.rd].i64 = bp[ins.r.rs1].i64 >> bp[ins.r.rs2].i64; break;

                case mod_u64: bp[ins.rd].u64 = bp[ins.r.rs1].u64 % bp[ins.r.rs2].u64; break;
                case mod_i64: bp[ins.rd].i64 = bp[ins.r.rs1].i64 % bp[ins.r.rs2].i64; break;

                case add_f64: bp[ins.rd].f64 = bp[ins.r.rs1].f64 + bp[ins.r.rs2].f64; break;
                case sub_f64: bp[ins.rd].f64 = bp[ins.r.rs1].f64 - bp[ins.r.rs2].f64; break;
                case mul_f64: bp[ins.rd].f64 = bp[ins.r.rs1].f64 * bp[ins.r.rs2].f64; break;
                case div_f64: bp[ins.rd].f64 = bp[ins.r.rs1].f64 / bp[ins.r.rs2].f64; break;

                case add_imm_i8: bp[ins.rd].u64 = bp[ins.i.rs1].u64 + (int64_t)ins.i.imm; break;
                case load_imm_i16: bp[ins.rd].i64 = ins.u.imm; break;

                case move_64: bp[ins.rd].u64 = (uint64_t)bp[ins.i.rs1].u64; break;

                case set_is_less_than_u64: bp[ins.rd].u64 = uint64_t(bp[ins.r.rs1].u64 < bp[ins.r.rs2].u64); break;
                case set_is_less_than_i64: bp[ins.rd].u64 = uint64_t(bp[ins.r.rs1].i64 < bp[ins.r.rs2].i64); break;
                case set_is_less_than_f64: bp[ins.rd].u64 = uint64_t(bp[ins.r.rs1].f64 < bp[ins.r.rs2].f64); break;
                case set_is_less_equal_u64: bp[ins.rd].u64 = uint64_t(bp[ins.r.rs1].u64 <= bp[ins.r.rs2].u64); break;
                case set_is_less_equal_i64: bp[ins.rd].u64 = uint64_t(bp[ins.r.rs1].i64 <= bp[ins.r.rs2].i64); break;
                case set_is_equal_64: bp[ins.rd].u64 = uint64_t(bp[ins.r.rs1].u64 == bp[ins.r.rs2].u64); break;
                case set_is_not_equal_64: bp[ins.rd].u64 = uint64_t(bp[ins.r.rs1].u64 != bp[ins.r.rs2].u64); break;
                
                case jump: pc += (std::ptrdiff_t)ins.u.imm; break;
                case jump_if_true: if(bp[ins.rd].u64 != 0) pc += (std::ptrdiff_t)ins.u.imm; break;
                case jump_if_false: if(bp[ins.rd].u64 == 0) pc += (std::ptrdiff_t)ins.u.imm; break;
                case jump_if_less_than_i64: if (bp[ins.rd].i64 < bp[ins.i.rs1].i64) pc += (std::ptrdiff_t)ins.i.imm; break;
                case jump_if_less_equal_i64: if (bp[ins.rd].i64 <= bp[ins.i.rs1].i64) pc += (std::ptrdiff_t)ins.i.imm; break;
                case loop_inc_check_jump: if (++bp[ins.rd].i64 < bp[ins.i.rs1].i64) [[unlikely]] pc -= (uint8_t)ins.i.imm; break;

                case load_64:  bp[ins.rd].u64 = ((const uint64_t*)std::assume_aligned<8>(bp[ins.r.rs1].ptr))[bp[ins.r.rs2].i64]; break;
                case load_u32: bp[ins.rd].u64 = ((const uint32_t*)std::assume_aligned<4>(bp[ins.r.rs1].ptr))[bp[ins.r.rs2].i64]; break;
                case load_i32: bp[ins.rd].i64 = ((const int32_t* )std::assume_aligned<4>(bp[ins.r.rs1].ptr))[bp[ins.r.rs2].i64]; break;
                case load_u8:  bp[ins.rd].u64 = ((const uint8_t* )std::assume_aligned<1>(bp[ins.r.rs1].ptr))[bp[ins.r.rs2].i64]; break;
                case load_i16: [[unlikely]] bp[ins.rd].i64 = ((const int16_t* )std::assume_aligned<2>(bp[ins.r.rs1].ptr))[bp[ins.r.rs2].i64]; break;
                case load_u16: [[unlikely]] bp[ins.rd].u64 = ((const uint16_t*)std::assume_aligned<2>(bp[ins.r.rs1].ptr))[bp[ins.r.rs2].i64]; break;
                case load_i8:  [[unlikely]] bp[ins.rd].i64 = ((const int8_t*  )std::assume_aligned<1>(bp[ins.r.rs1].ptr))[bp[ins.r.rs2].i64]; break;

                case store_64: ((uint64_t*)std::assume_aligned<8>(bp[ins.r.rs1].ptr))[bp[ins.r.rs2].i64] = bp[ins.rd].u64; break;
                case store_32: ((uint32_t*)std::assume_aligned<4>(bp[ins.r.rs1].ptr))[bp[ins.r.rs2].i64] = bp[ins.rd].u32; break;
                case store_8:  ((uint8_t* )std::assume_aligned<1>(bp[ins.r.rs1].ptr))[bp[ins.r.rs2].i64] = bp[ins.rd].u8; break;
                case store_16: [[unlikely]] ((uint16_t*)std::assume_aligned<2>(bp[ins.r.rs1].ptr))[bp[ins.r.rs2].i64] = bp[ins.rd].u16; break;

                case load_global_64: [[unlikely]] bp[ins.rd].u64 = value_stack[(uint16_t)ins.u.imm].u64; break;
                case store_global_64: [[unlikely]] value_stack[(uint16_t)ins.u.imm].u64 = bp[ins.rd].u64; break;
                case load_constant_str: [[unlikely]] bp[ins.rd].ptr = (uint64_t*)exe.constants.data() + (uint16_t)ins.u.imm; break;
                case load_constant_64: [[unlikely]] bp[ins.rd].u64 = *((uint64_t*)exe.constants.data() + (uint16_t)ins.u.imm);  break;
                
                case cast_i64_to_f64: [[unlikely]] bp[ins.rd].f64 = static_cast<double>(bp[ins.i.rs1].i64); break;
                case cast_f64_to_i64: [[unlikely]] bp[ins.rd].i64 = static_cast<int64_t>(bp[ins.i.rs1].f64); break;
                        
                case call:
                    *(sp++) = ins.rd;
                    *(sp++) = (uint64_t)pc;
                    bp += ins.rd;
                    pc += (std::ptrdiff_t)ins.u.imm;
                    break;
                    
                case call_reg: [[unlikely]]
                    *(sp++) = (uint16_t)ins.u.imm;
                    *(sp++) = (uint64_t)pc;
                    bp += (uint16_t)ins.u.imm;
                    pc = &exe.bytecodes[bp[ins.rd].u32];
                    break;
                    
                case call_host: {
                    const uint16_t function_id = (uint16_t)ins.u.imm;
                    host_functions[function_id].second(bp + ins.rd, bp + ins.rd);
                    break;
                }
                
                case ret:
                    pc = std::assume_aligned<4>((instruction_t*)*(--sp));
                    bp -= *(--sp);
                    break;
                
                case nop: [[unlikely]] break;
                case halt: [[unlikely]] goto end;

                default: [[unlikely]] std::unreachable();
            }
        }
    end:
        this->bp = bp;
        this->sp = sp;
        this->pc = pc;
        this->is_running = false;
    }

};
}