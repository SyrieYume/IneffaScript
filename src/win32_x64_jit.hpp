#pragma once
#ifndef _WIN32
#error "This JIT compiler only supports Windows"
#endif

static_assert(sizeof(void*) == 8, "This JIT compiler only supports 64-bit architecture (x64)");
#if !defined(_M_X64) && !defined(__x86_64__)
    static_assert(false, "x64 architecture macro not detected");
#endif

#include <format>
#include <stdexcept>
#include <vector>
#include <span>
#include <cstring>

#ifndef NOMINMAX
    #define NOMINMAX
#endif

#include <windows.h>
#include "bytecode.hpp"

namespace ineffa::script {

class x64_assember_t {
public:
    enum class reg_t : uint8_t {
        NO_REG = 0,
        RAX=0, RCX=1, RDX=2, RBX=3, RSP=4, RBP=5, RSI=6, RDI=7,
        R8=8, R9=9, R10=10, R11=11, R12=12, R13=13, R14=14, R15=15
    };

    enum class op_t : uint8_t {
        ADD_R_RM  = 0x03,
        OR_R_RM   = 0x0B,
        AND_R_RM  = 0x23,
        SUB_R_RM  = 0x2B,
        XOR_R_RM  = 0x33,
        CMP_R_RM  = 0x3B,
        
        ALU_IMM8  = 0x83, 
        TEST_RM_R = 0x85,
        
        MOV_RM8_R8   = 0x88,
        MOV_RM_R     = 0x89,
        MOV_R_RM     = 0x8B,
        MOVSXD       = 0x63,
        LEA          = 0x8D,
        MOV_RM_IMM32 = 0xC7,
        MOV_IMM_REG_BASE = 0xB8,

        SHIFT_CL    = 0xD3,

        CALL_REL    = 0xE8,
        JMP_REL     = 0xE9,
        
        // Group 3 - NEG, NOT, MUL, DIV
        GRP3_64     = 0xF7,
        
        // Group 5 - INC, DEC, CALL rm, JMP rm, PUSH rm
        GRP5        = 0xFF,

        PUSH_R_BASE = 0x50,
        POP_R_BASE  = 0x58,
        RET         = 0xC3,
        
        PRE_SSE_66  = 0x66,
        PRE_SSE_F2  = 0xF2,
        
        EXT_JE      = 0x84,
        EXT_JNE     = 0x85,
        EXT_JL      = 0x8C,
        EXT_JLE     = 0x8E,

        EXT_SET_E   = 0x94,
        EXT_SET_NE  = 0x95,
        EXT_SET_L   = 0x9C,
        EXT_SET_B   = 0x92,
        EXT_SET_LE  = 0x9E,
        EXT_SET_BE  = 0x96,
        
        EXT_IMUL    = 0xAF,

        EXT_MOVZX_R8  = 0xB6,
        EXT_MOVZX_R16 = 0xB7,
        EXT_MOVSX_R8  = 0xBE,
        EXT_MOVSX_R16 = 0xBF,

        EXT_CVT_I2D = 0x2A,
        EXT_CVT_D2I = 0x2C,
        EXT_ADD_SD  = 0x58,
        EXT_MUL_SD  = 0x59,
        EXT_SUB_SD  = 0x5C,
        EXT_DIV_SD  = 0x5E,
        EXT_MOV_D2Q = 0x7E,
        EXT_MOV_Q2D = 0x6E,
    };

    enum class mod_r_m_t : uint8_t {
        Mod_Mem = 0,     // [reg]
        Mod_Disp8 = 1,   // [reg + 8bit]
        Mod_Disp32 = 2,  // [reg + 32bit]
        Mod_Reg = 3      // reg
    };
  
    enum class op_ext_t : uint8_t {
        ADD = 0, OR = 1, ADC = 2, SBB = 3, AND = 4, SUB = 5, XOR = 6, CMP = 7,  // ALU 扩展码
        ROL = 0, ROR = 1, RCL = 2, RCR = 3, SHL = 4, SHR = 5, SAR = 7,          // Shift 扩展码
        TEST_IMM = 0, NOT = 2, NEG = 3, MUL = 4, IMUL = 5, DIV = 6, IDIV = 7    // Group 3 扩展码
    };

    using enum reg_t;
    using enum op_t;
    using enum mod_r_m_t;

    static constexpr reg_t bp_reg = reg_t::R13;
    std::vector<uint8_t> codes;

    template <typename... Args>
    void emit(Args&&... args) {
        (codes.push_back((uint8_t)args), ...);
    }

    void emit_u32(uint32_t u) { 
        size_t sz = codes.size();
        codes.resize(sz + 4);
        std::memcpy(codes.data() + sz, &u, 4);
    }

    void emit_u64(uint64_t u) {
        size_t sz = codes.size();
        codes.resize(sz + 8);
        std::memcpy(codes.data() + sz, &u, 8);
    }

    void emit_rex(reg_t reg, reg_t base, reg_t index = RAX, bool is_64bit = true) {
        uint8_t rex = 0x40 | (is_64bit << 3) | (((uint8_t)reg & 8) >> 1) | (((uint8_t)index & 8) >> 2) | (((uint8_t)base & 8) >> 3);
        if (rex != 0x40 || (reg >= RSP && reg <= RDI && !is_64bit)) emit(rex);
    }

    template<typename T>
    requires std::same_as<T, reg_t> || std::same_as<T, op_ext_t>
    void emit_modrm(mod_r_m_t mod, T reg_or_opcode_ext, reg_t base) {
        uint8_t modrm = (((uint8_t)mod & 3) << 6) | (((uint8_t)reg_or_opcode_ext & 7) << 3) | ((uint8_t)base & 7);
        emit(modrm);
    }

    void emit_alu64(op_t opcode, reg_t dst, reg_t src) {
        emit_rex(dst, src);
        emit(opcode);
        emit_modrm(Mod_Reg, dst, src);
    }

    void emit_mov(reg_t dst_idx, reg_t src_idx) {
        if (dst_idx == src_idx) return;
        emit_alu64(MOV_R_RM, dst_idx, src_idx);
    }

    void emit_alu64_imm8(op_ext_t opcode_ext, reg_t dst, int8_t imm) {
        emit_rex(NO_REG, dst);
        emit(ALU_IMM8);
        emit_modrm(Mod_Reg, opcode_ext, dst); 
        emit((uint8_t)imm);
    }

    void emit_div64(reg_t dst, reg_t lhs, reg_t rhs, bool is_signed, bool is_mod) {
        emit_mov(RAX, lhs);

        if (is_signed) {
            emit_rex(NO_REG, NO_REG);
            emit(0x99); // CDQ / CQO
        } else
            emit_alu64(XOR_R_RM, RDX, RDX);

        emit_rex(NO_REG, rhs);
        emit(GRP3_64); 
        emit_modrm(Mod_Reg, is_signed ? op_ext_t::IDIV : op_ext_t::DIV, rhs);
        emit_mov(dst, is_mod ? RDX : RAX);
    }

    void emit_shift64(op_ext_t opcode_ext, reg_t dst, reg_t count_src) {
        emit_mov(RCX, count_src);
        emit_rex(NO_REG, dst);
        emit(SHIFT_CL);
        emit_modrm(Mod_Reg, opcode_ext, dst);
    }

    void emit_f64_op(op_t opcode, reg_t dst, reg_t rhs) {
        reg_t xmm0 = reg_t(0), xmm1 = reg_t(1);

        // MOVQ xmm0, dst
        emit(PRE_SSE_66); emit_rex(xmm0, dst); emit(0x0F, EXT_MOV_Q2D); emit_modrm(Mod_Reg, xmm0, dst); 
        
        // MOVQ xmm1, rhs
        emit(PRE_SSE_66); emit_rex(xmm1, rhs); emit(0x0F, EXT_MOV_Q2D); emit_modrm(Mod_Reg, xmm1, rhs);

        // OP xmm0, xmm1
        emit(PRE_SSE_F2); emit_rex(xmm0, xmm1); emit(0x0F, opcode); emit_modrm(Mod_Reg, xmm0, xmm1);

        // MOVQ dst, xmm0
        emit(PRE_SSE_66); emit_rex(xmm0, dst); emit(0x0F, EXT_MOV_D2Q); emit_modrm(Mod_Reg, xmm0, dst); 
    }

    void emit_cmp_set(op_t opcode, reg_t dst, reg_t lhs, reg_t rhs) {
        // CMP lhs, rhs
        emit_alu64(CMP_R_RM, lhs, rhs);

         // SETcc
        emit_rex(NO_REG, dst);
        emit(0x0F, opcode);
        emit_modrm(Mod_Reg, NO_REG, dst);

        // MOVZX dst, dst_byte
        emit_rex(dst, dst);
        emit(0x0F, EXT_MOVZX_R8);
        emit_modrm(Mod_Reg, dst, dst);
    }

    void emit_push(reg_t reg) {
        if (uint8_t(reg) >= 8)  emit(0x41); // REX.B
        emit(uint8_t(PUSH_R_BASE) | (uint8_t(reg) & 7));
    }

    void emit_pop(reg_t reg) {
        if (uint8_t(reg) >= 8)  emit(0x41); // REX.B
        emit(uint8_t(POP_R_BASE) | (uint8_t(reg) & 7));
    }

    void emit_jit_func_start() {
        emit_push(RBX);
        emit_push(RDI);
        emit_push(RSI);
        emit_push(R12);
        emit_push(R13);
        emit_push(R14);
        emit_push(R15);
        emit(0x48, 0x83, 0xEC, 0x20);  // SUB RSP, 32
        emit_mov(bp_reg, RCX);
    }

    
    void emit_jit_func_end() {
        emit(0x48, 0x83, 0xC4, 0x20);  // ADD RSP, 32
        emit_pop(R15);
        emit_pop(R14);
        emit_pop(R13);
        emit_pop(R12);
        emit_pop(RSI);
        emit_pop(RDI);
        emit_pop(RBX);
        emit(RET);
    }

    void emit_reg_mem_op(op_t opcode_base, reg_t reg, reg_t base, int32_t offset) {
        emit_rex(reg, bp_reg);
        emit(opcode_base);

        if (std::in_range<int8_t>(offset)) {
            emit_modrm(Mod_Disp8, reg, base);
            emit(uint8_t(offset));
        } else {
            emit_modrm(Mod_Disp32, reg, base);
            emit_u32(uint32_t(offset));
        }
    }

    void load_from_bp_index(reg_t reg, uint8_t index) { 
        emit_reg_mem_op(MOV_R_RM, reg, bp_reg, (int32_t)index * 8); 
    }

    void store_to_bp_index(reg_t reg, uint8_t index) noexcept { 
        emit_reg_mem_op(MOV_RM_R, reg, bp_reg, (int32_t)index * 8); 
    }

    void load_imm(reg_t dst, int64_t value) noexcept {
        emit_rex(NO_REG, dst);
        if (std::in_range<int32_t>(value)) {
            emit(MOV_RM_IMM32);
            emit_modrm(Mod_Reg, NO_REG, dst);
            emit_u32((uint32_t)(int32_t)value);
        } 
        else {
            emit((uint8_t)MOV_IMM_REG_BASE | ((uint8_t)dst & 7));
            emit_u64(value);
        }
    }

    void emit_mem_sib(bool is_store, int width_bytes, bool is_signed, reg_t src_or_dst, reg_t base, reg_t index) {
        assert(index != RSP && "RSP cannot be used as an index register in SIB encoding");

        uint8_t scale = (width_bytes == 8) ? 3 : (width_bytes == 4) ? 2 : (width_bytes == 2) ? 1 : 0;

        if (width_bytes == 2) 
            emit(0x66);

        bool is_64bit = is_store ? 
            (width_bytes == 8) : 
            (width_bytes == 8 || (width_bytes == 4 && is_signed));

        emit_rex(src_or_dst, base, index, is_64bit);

        if (is_store)
            emit(width_bytes == 1 ? MOV_RM8_R8 : MOV_RM_R);

        else switch (width_bytes) {
            case 8: emit(MOV_R_RM); break;
            case 4: emit(is_signed ? MOVSXD : MOV_R_RM); break;
            case 2: emit(0x0F, is_signed ? EXT_MOVSX_R16 : EXT_MOVZX_R16); break;
            case 1: emit(0x0F, is_signed ? EXT_MOVSX_R8 : EXT_MOVZX_R8); break;
        }

        emit_modrm(Mod_Mem, src_or_dst, RSP); 

        uint8_t sib = (scale << 6) | ((uint8_t(index) & 7) << 3) | (uint8_t(base) & 7);
        emit(sib);
    }
};


class x64_register_allocator_t {
private:
    x64_assember_t& as; 
    uint32_t current_tick = 0;

public:
    using enum x64_assember_t::reg_t;
    static constexpr std::array allocable_regs = { RBX, RSI, RDI, R8, R9, R10, R11, R14, R15 };

    struct reg_state_t {
        uint32_t last_access_time;
        uint8_t bp_reg_idx;
        bool is_valid;
        bool is_dirty;
    };

    std::array<reg_state_t, sizeof(allocable_regs)> regs_state = {};

    x64_register_allocator_t(x64_assember_t& assember) : as(assember) {}

    void setup_reg(uint32_t i, uint8_t bp_reg_idx, bool is_to_write) {
        regs_state[i] = {
            .last_access_time = current_tick,
            .bp_reg_idx = bp_reg_idx,
            .is_valid = true,
            .is_dirty = is_to_write
        };
        if (!is_to_write)
            as.load_from_bp_index(allocable_regs[i], bp_reg_idx);
    }

    void fflush_reg(uint32_t reg_index) {
        if (!regs_state[reg_index].is_valid)
            return;
        if (regs_state[reg_index].is_dirty)
            as.store_to_bp_index(allocable_regs[reg_index], regs_state[reg_index].bp_reg_idx);
        regs_state[reg_index].is_dirty = false;
    }

    auto alloc(uint8_t bp_reg_idx, bool is_to_write) -> x64_assember_t::reg_t {
        current_tick++;

        for (uint32_t i = 0; i < allocable_regs.size(); i++)
            if (regs_state[i].is_valid && regs_state[i].bp_reg_idx == bp_reg_idx) {
                regs_state[i].last_access_time = current_tick;
                if (is_to_write) regs_state[i].is_dirty = true;
                return allocable_regs[i];
            }

        for (uint32_t i = 0; i < allocable_regs.size(); i++)
            if (!regs_state[i].is_valid) {
                setup_reg(i, bp_reg_idx, is_to_write);
                return allocable_regs[i];
            }

        uint32_t min_tick = std::numeric_limits<uint32_t>::max();
        int32_t eviction_idx = 0;
        
        for (uint32_t i = 0; i < allocable_regs.size(); i++) {
            if (regs_state[i].last_access_time < min_tick) {
                min_tick = regs_state[i].last_access_time;
                eviction_idx = i;
            }
        }
        
        fflush_reg(eviction_idx);
        setup_reg(eviction_idx, bp_reg_idx, is_to_write);
        return allocable_regs[eviction_idx];
    }

    void mark_all_invalid() {
        for (uint32_t i = 0; i < allocable_regs.size(); i++)
            regs_state[i].is_valid = false;
    }

    void mark_dirty(x64_assember_t::reg_t reg) {
        for (uint32_t i = 0; i < allocable_regs.size(); i++) {
            if (allocable_regs[i] == reg) {
                regs_state[i].is_dirty = true;
                break;
            }
        }
    }

    void fflush_all(std::array<uint8_t, 2> bp_regs_fflush_range = {0, 255}) {
        for (uint32_t i = 0; i < allocable_regs.size(); i++)
            if (uint8_t bp_reg_idx = regs_state[i].bp_reg_idx; bp_reg_idx >= bp_regs_fflush_range[0] && bp_reg_idx < bp_regs_fflush_range[1])
                fflush_reg(i);
    }
};


class win32_x64_jit {
private:
    struct bytecode_info_t {
        uint32_t target_code_offset;
        bool is_function_start = false;
        bool is_jump_target = false;
    };

    struct jump_patch_t {
        uint32_t code_offset;
        uint32_t source_bytecode_index;
        int32_t target_bytecode_diff;
    };

public:
    struct jit_func_t {
    public:
        void operator()(value_t* bp) const {
            auto func = (void(*)(value_t*))exe_mem;
            func(bp);
        }

        explicit jit_func_t(std::span<uint8_t> codes) {
            if ((exe_mem = VirtualAlloc(NULL, codes.size(), MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE)) == nullptr) 
                throw std::system_error(GetLastError(), std::system_category(), "VirtualAlloc failed");

            std::memcpy(exe_mem, codes.data(), codes.size());

            if (DWORD old_protect; !VirtualProtect(exe_mem, codes.size(), PAGE_EXECUTE_READ, &old_protect)) {
                VirtualFree(exe_mem, 0, MEM_RELEASE);
                throw std::system_error(GetLastError(), std::system_category(), "VirtualProtect failed");
            }

            FlushInstructionCache(GetCurrentProcess(), exe_mem, codes.size());
        }

        ~jit_func_t() noexcept {
            if (exe_mem) VirtualFree(exe_mem, 0, MEM_RELEASE);
        }

        jit_func_t(const jit_func_t&) = delete;
        jit_func_t& operator=(const jit_func_t&) = delete;
        jit_func_t(jit_func_t&& other) noexcept : exe_mem(std::exchange(other.exe_mem, nullptr)) {}
        
        jit_func_t& operator=(jit_func_t&& other) noexcept {
            return (std::swap(exe_mem, other.exe_mem), *this);
        }

    private:
        void *exe_mem;
    };


    static auto compile(const executable_t& exe, std::span<std::pair<std::string_view, host_func_t>> host_functions) -> jit_func_t {
        using as_t = x64_assember_t;
        using enum x64_assember_t::op_t;
        using enum x64_assember_t::mod_r_m_t;
        
        x64_assember_t as;
        auto regs = x64_register_allocator_t(as);
        std::vector<bytecode_info_t> bytecodes_info;
        bytecodes_info.resize(exe.bytecodes.size() + 1);
        std::vector<jump_patch_t> patches;

        for (auto func : exe.functions)
            bytecodes_info[func.start].is_function_start = true;
            
        for (uint32_t i = 0; i < exe.bytecodes.size(); ++i) {
            const auto& ins = exe.bytecodes[i];
            int32_t target = -1;

            switch (ins.opcode) {
                case opcode_t::jump:
                case opcode_t::jump_if_true:
                case opcode_t::jump_if_false:
                    target = i + 1 + (int32_t)ins.u.imm;
                    break;
                case opcode_t::jump_if_less_than_i64:
                case opcode_t::jump_if_less_equal_i64:
                    target = i + 1 + (int32_t)ins.i.imm;
                    break;
                case opcode_t::loop_inc_check_jump:
                    target = i + 1 - (int32_t)(uint8_t)ins.i.imm;
                    break;
                default:
                    break;
            }

            if (target >= 0 && target < (int32_t)bytecodes_info.size())
                bytecodes_info[target].is_jump_target = true;
        }
        
        as.emit_jit_func_start();

        for (auto [i, ins] : exe.bytecodes | std::views::enumerate) {
            if (bytecodes_info[i].is_jump_target) {
                regs.fflush_all();
                regs.mark_all_invalid();
            }

            bytecodes_info[i].target_code_offset = as.codes.size();

            if (bytecodes_info[i].is_function_start)
                regs.mark_all_invalid();
            
            switch (ins.opcode) {
                case opcode_t::add_64:
                case opcode_t::sub_64: 
                case opcode_t::mul_64:
                case opcode_t::div_u64:
                case opcode_t::mod_u64:
                case opcode_t::div_i64: 
                case opcode_t::mod_i64:
                case opcode_t::and_64:
                case opcode_t::or_64:
                case opcode_t::xor_64:
                case opcode_t::shl_64:
                case opcode_t::shr_u64:
                case opcode_t::shr_i64:
                case opcode_t::add_f64:
                case opcode_t::sub_f64:
                case opcode_t::mul_f64:
                case opcode_t::div_f64:
                case opcode_t::set_is_less_than_u64:
                case opcode_t::set_is_less_than_i64:
                case opcode_t::set_is_less_equal_u64:
                case opcode_t::set_is_less_equal_i64:
                case opcode_t::set_is_equal_64:
                case opcode_t::set_is_not_equal_64:
                {
                    auto rs1 = regs.alloc(ins.r.rs1, false);
                    auto rs2 = regs.alloc(ins.r.rs2, false);
                    auto rd = regs.alloc(ins.rd, true);

                    switch (ins.opcode) {
                        case opcode_t::add_64:   as.emit_mov(rd, rs1); as.emit_alu64(ADD_R_RM, rd, rs2); break;
                        case opcode_t::sub_64:   as.emit_mov(rd, rs1); as.emit_alu64(SUB_R_RM, rd, rs2); break;
                        case opcode_t::and_64:   as.emit_mov(rd, rs1); as.emit_alu64(AND_R_RM, rd, rs2); break;
                        case opcode_t::or_64:    as.emit_mov(rd, rs1); as.emit_alu64(OR_R_RM, rd, rs2); break;
                        case opcode_t::xor_64:   as.emit_mov(rd, rs1); as.emit_alu64(XOR_R_RM, rd, rs2); break;

                        case opcode_t::shl_64:   as.emit_mov(rd, rs1); as.emit_shift64(as_t::op_ext_t::SHL, rd, rs2); break;
                        case opcode_t::shr_u64:  as.emit_mov(rd, rs1); as.emit_shift64(as_t::op_ext_t::SHR, rd, rs2); break;
                        case opcode_t::shr_i64:  as.emit_mov(rd, rs1); as.emit_shift64(as_t::op_ext_t::SAR, rd, rs2); break;
                        case opcode_t::add_f64:  as.emit_mov(rd, rs1); as.emit_f64_op(EXT_ADD_SD, rd, rs2); break;
                        case opcode_t::sub_f64:  as.emit_mov(rd, rs1); as.emit_f64_op(EXT_SUB_SD, rd, rs2); break;
                        case opcode_t::mul_f64:  as.emit_mov(rd, rs1); as.emit_f64_op(EXT_MUL_SD, rd, rs2); break;
                        case opcode_t::div_f64:  as.emit_mov(rd, rs1); as.emit_f64_op(EXT_DIV_SD, rd, rs2); break;

                        case opcode_t::div_u64:  as.emit_div64(rd, rs1, rs2, false, false); break;
                        case opcode_t::mod_u64:  as.emit_div64(rd, rs1, rs2, false, true); break;
                        case opcode_t::div_i64:  as.emit_div64(rd, rs1, rs2, true, false); break;
                        case opcode_t::mod_i64:  as.emit_div64(rd, rs1, rs2, true, true); break;
                        case opcode_t::set_is_equal_64: as.emit_cmp_set(EXT_SET_E, rd, rs1, rs2); break;
                        case opcode_t::set_is_not_equal_64: as.emit_cmp_set(EXT_SET_NE, rd, rs1, rs2); break;
                        case opcode_t::set_is_less_than_i64: as.emit_cmp_set(EXT_SET_L, rd, rs1, rs2); break;
                        case opcode_t::set_is_less_equal_i64: as.emit_cmp_set(EXT_SET_LE, rd, rs1, rs2); break;
                        case opcode_t::set_is_less_than_u64: as.emit_cmp_set(EXT_SET_B, rd, rs1, rs2); break;
                        case opcode_t::set_is_less_equal_u64: as.emit_cmp_set(EXT_SET_BE, rd, rs1, rs2); break;

                        case opcode_t::mul_64: 
                            as.emit_mov(rd, rs1);
                            as.emit_rex(rd, rs2);
                            as.emit(0x0F, EXT_IMUL);
                            as.emit_modrm(Mod_Reg, rd, rs2);
                            break;
                        
                        default: std::unreachable();
                    }
                    break;
                }

                case opcode_t::not_64: {
                    auto rs1 = regs.alloc(ins.r.rs1, false);
                    auto rd = regs.alloc(ins.rd, true);
                    as.emit_mov(rd, rs1);
                    as.emit_rex(rd, as_t::NO_REG);
                    as.emit(as_t::op_t::GRP3_64);
                    as.emit_modrm(Mod_Reg, as_t::op_ext_t::NOT, rd);
                    break;
                }

                case opcode_t::add_imm_i8: {
                    auto rs1 = regs.alloc(ins.i.rs1, false);
                    auto rd = regs.alloc(ins.rd, true);
                    as.emit_mov(rd, rs1);
                    as.emit_alu64_imm8(as_t::op_ext_t::ADD, rd, ins.i.imm);
                    break;
                }
                
                case opcode_t::move_64: {
                    auto rs1 = regs.alloc(ins.i.rs1, false);
                    auto rd = regs.alloc(ins.rd, true);
                    as.emit_mov(rd, rs1);
                    break;
                }

                case opcode_t::load_imm_i16: {
                    auto rd = regs.alloc(ins.rd, true);
                    as.load_imm(rd, (int64_t)ins.u.imm);
                    break;
                }
                
                case opcode_t::cast_i64_to_f64: {
                    auto rs1 = regs.alloc(ins.i.rs1, false);
                    auto rd = regs.alloc(ins.rd, true);
                    auto xmm0 = as_t::reg_t(0);

                    // CVTSI2SD xmm0, rs1
                    as.emit(PRE_SSE_F2); 
                    as.emit_rex(xmm0, rs1); 
                    as.emit(0x0F, EXT_CVT_I2D); 
                    as.emit_modrm(Mod_Reg, xmm0, rs1);

                    // MOVQ rd, xmm0
                    as.emit(PRE_SSE_66); 
                    as.emit_rex(xmm0, rd);
                    as.emit(0x0F, EXT_MOV_D2Q); 
                    as.emit_modrm(Mod_Reg, xmm0, rd);
                    break;
                }
                
                case opcode_t::cast_f64_to_i64: {
                    auto rs1 = regs.alloc(ins.i.rs1, false);
                    auto rd = regs.alloc(ins.rd, true);
                    auto xmm0 = as_t::reg_t(0);

                    // MOVQ xmm0, rs1
                    as.emit(PRE_SSE_66); 
                    as.emit_rex(xmm0, rs1);
                    as.emit(0x0F, EXT_MOV_Q2D);
                    as.emit_modrm(Mod_Reg, xmm0, rs1);

                    // CVTTSD2SI rd, xmm0
                    as.emit(PRE_SSE_F2); 
                    as.emit_rex(rd, xmm0); 
                    as.emit(0x0F, EXT_CVT_D2I); 
                    as.emit_modrm(Mod_Reg, rd, xmm0);
                    break;
                }

                case opcode_t::jump: {
                    regs.fflush_all();
                    
                    // JMP rel32
                    as.emit(JMP_REL);
                    patches.push_back({ (uint32_t)as.codes.size(), (uint32_t)i, (int32_t)ins.u.imm });
                    as.emit_u32(0);
                    break;
                }

                case opcode_t::jump_if_true:
                case opcode_t::jump_if_false: {
                    auto rd = regs.alloc(ins.rd, false);

                    regs.fflush_all();

                    // TEST rd, rd
                    as.emit_rex(rd, rd);
                    as.emit(TEST_RM_R);
                    as.emit_modrm(Mod_Reg, rd, rd);
                    
                    // JNE / JE rel32
                    as.emit(0x0F, ins.opcode == opcode_t::jump_if_true ? EXT_JNE : EXT_JE);
                    patches.push_back({ (uint32_t)as.codes.size(), (uint32_t)i, (int32_t)ins.u.imm });
                    as.emit_u32(0);
                    break;
                }

                case opcode_t::jump_if_less_than_i64:
                case opcode_t::jump_if_less_equal_i64: {
                    auto rd = regs.alloc(ins.rd, false);
                    auto rs1 = regs.alloc(ins.i.rs1, false);

                    regs.fflush_all();

                    // CMP rd, rs1
                    as.emit_alu64(CMP_R_RM, rd, rs1);

                    // JL / JLE rel32
                    as.emit(0x0F, ins.opcode == opcode_t::jump_if_less_than_i64 ? EXT_JL : EXT_JLE);
                    patches.push_back({ (uint32_t)as.codes.size(), (uint32_t)i, (int32_t)ins.i.imm });
                    as.emit_u32(0);
                    break;
                }

                case opcode_t::loop_inc_check_jump: {
                    auto rs1 = regs.alloc(ins.i.rs1, false);
                    auto rd = regs.alloc(ins.rd, false);
                    regs.mark_dirty(rd);

                    // INC r64
                    as.emit_rex(as_t::NO_REG, rd);
                    as.emit(as_t::GRP5);
                    as.emit_modrm(Mod_Reg, as_t::NO_REG, rd);

                    regs.fflush_all();

                    // CMP rd, rs1
                    as.emit_alu64(CMP_R_RM, rd, rs1);

                    // JL rel32
                    as.emit(0x0F, EXT_JL);
                    patches.push_back({ (uint32_t)as.codes.size(), (uint32_t)i, -(int32_t)ins.i.imm });
                    as.emit_u32(0);
                    break;
                }

                case opcode_t::load_64:
                case opcode_t::load_u32:
                case opcode_t::load_i32:
                case opcode_t::load_u16: 
                case opcode_t::load_i16: 
                case opcode_t::load_u8:  
                case opcode_t::load_i8:
                case opcode_t::store_64:
                case opcode_t::store_32:  
                case opcode_t::store_16:
                case opcode_t::store_8: {
                    struct { int width_bytes; bool is_signed; bool is_store; } cfg;
                    switch (ins.opcode) {
                        case opcode_t::store_8:  cfg = {1, 0, 1}; break;
                        case opcode_t::store_16: cfg = {2, 0, 1}; break;
                        case opcode_t::store_32: cfg = {4, 0, 1}; break;
                        case opcode_t::store_64: cfg = {8, 0, 1}; break;
                        case opcode_t::load_u8:  cfg = {1, 0, 0}; break;
                        case opcode_t::load_i8:  cfg = {1, 1, 0}; break;
                        case opcode_t::load_u16: cfg = {2, 0, 0}; break;
                        case opcode_t::load_i16: cfg = {2, 1, 0}; break;
                        case opcode_t::load_u32: cfg = {4, 0, 0}; break;
                        case opcode_t::load_i32: cfg = {4, 1, 0}; break;
                        case opcode_t::load_64:  cfg = {8, 0, 0}; break;
                        default: std::unreachable();
                    }

                    auto base  = regs.alloc(ins.r.rs1, false);
                    auto index = regs.alloc(ins.r.rs2, false);
                    auto src_or_dst = regs.alloc(ins.rd, !cfg.is_store);

                    as.emit_mem_sib(cfg.is_store, cfg.width_bytes, cfg.is_signed, src_or_dst, base, index);
                    break;
                }

                case opcode_t::call: {
                    regs.fflush_all();
                    regs.mark_all_invalid();

                    // PUSH R13
                    as.emit_push(as_t::bp_reg);

                    // LEA R13, [R13 + ins.rd * 8]
                    as.emit_reg_mem_op(LEA, as_t::bp_reg, as_t::bp_reg, (int32_t)ins.rd * 8);

                    // CALL rel32
                    as.emit(CALL_REL);
                    patches.push_back({ (uint32_t)as.codes.size(), (uint32_t)i, (int32_t)ins.u.imm });
                    as.emit_u32(0);

                    // POP R13
                    as.emit_pop(as.bp_reg);
                    break;
                }

                case opcode_t::ret: {
                    auto it = std::find_if(exe.functions.begin(), exe.functions.end(), [=](const function_info_t& func) { 
                        return i >= func.start && i < func.end;
                    });

                    std::array<uint8_t, 2> bp_regs_ffush_range = { 0, it == exe.functions.end() ? uint8_t(255) : uint8_t(((*it).ret_val_size + 7) / 8) };
                    regs.fflush_all(bp_regs_ffush_range);

                    as.emit(RET); // RET
                    break;
                }

                case opcode_t::call_host: {
                    regs.fflush_all();
                    regs.mark_all_invalid();

                    as.emit(0x48, 0x83, 0xEC, 0x20); // SUB RSP, 32

                    // LEA RCX, [R13 + ins.rd * 8]
                    as.emit_reg_mem_op(LEA, as_t::RCX, as_t::bp_reg, ins.rd * 8);
                    
                    // MOV RDX, RCX
                    as.emit_mov(as_t::RDX, as_t::RCX);

                    // RAX = func_ptr
                    uint64_t func_ptr = (uint64_t)host_functions[ins.u.imm].second;
                    as.load_imm(as_t::RAX, func_ptr);
                    
                    // Call RAX
                    as.emit_rex(as.NO_REG, as_t::RAX);
                    as.emit(GRP5);
                    as.emit_modrm(Mod_Reg, as_t::op_ext_t(2), as_t::RAX);
                    
                    as.emit(0x48, 0x83, 0xC4, 0x20); // ADD RSP, 32
                    break;
                }

                case opcode_t::load_constant_64: {
                    auto rd = regs.alloc(ins.rd, true);
                    as.load_imm(rd, ((uint64_t*)exe.constants.data())[(uint16_t)ins.u.imm]);
                    break;
                }

                case opcode_t::load_constant_str: {
                    auto rd = regs.alloc(ins.rd, true);
                    const uint64_t* const_base = (const uint64_t*)exe.constants.data();
                    uint64_t target_addr = (uint64_t)(const_base + ins.u.imm);
                    as.load_imm(rd, target_addr);
                    break;
                }

                case opcode_t::halt:
                    as.emit_jit_func_end();
                    break;

                default: 
                    throw std::runtime_error(std::format("unsupported instruction: {}", opcode_to_string_array[(uint8_t)ins.opcode]));
            }
        }

        bytecodes_info[exe.bytecodes.size()].target_code_offset = as.codes.size();

        for (const auto& patch : patches) {
            size_t target_bytecode_idx = patch.source_bytecode_index + 1 + patch.target_bytecode_diff;
            uint32_t target_code_offset = bytecodes_info[target_bytecode_idx].target_code_offset;
            int32_t rel = (int32_t)(target_code_offset - patch.code_offset - 4);
            std::memcpy(as.codes.data() + patch.code_offset, &rel, sizeof(rel));
        }

        return jit_func_t(as.codes);
    }
};
}


[[maybe_unused]] static inline auto win32_x64_jit_example(const ineffa::script::executable_t& exe) {
    using namespace ineffa::script;

    auto value_stack = std::vector<value_t>(8 * 1024);
    std::vector<std::pair<std::string_view, host_func_t>> host_functions = {};

    win32_x64_jit::jit_func_t jit_func = win32_x64_jit::compile(exe, host_functions);
    jit_func(value_stack.data());
}