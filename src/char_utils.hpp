#pragma once
#include <array>

// 字符判断工具
// ps: 标准库的 isspace, isalpha 有坑
namespace char_utils {
    enum char_type_t : uint8_t {
        None   = 0,
        Space  = 1 << 0, // \t, \n, \r, \f, \v, space
        Digit  = 1 << 1, // 0-9
        Upper  = 1 << 2, // A-Z
        Lower  = 1 << 3, // a-z
        Under  = 1 << 4, // _
        Alpha  = Upper | Lower,
        Alnum  = Alpha | Digit,
        Ident  = Alpha | Digit | Under
    };

    static constexpr std::array<uint8_t, 256> table_ = []() -> std::array<uint8_t, 256> {
        std::array<uint8_t, 256> t = {};
        for (int i = 0; i < 256; ++i) {
            if (i == ' ' || i == '\t' || i == '\n' || i == '\r' || i == '\f' || i == '\v') t[i] |= Space;
            if (i >= '0' && i <= '9') t[i] |= Digit;
            if (i >= 'A' && i <= 'Z') t[i] |= Upper;
            if (i >= 'a' && i <= 'z') t[i] |= Lower;
            if (i == '_') t[i] |= Under;
        }
        return t;
    }();

    static inline bool is_space(char c) noexcept {
        return table_[static_cast<uint8_t>(c)] & char_type_t::Space;
    }

    static inline bool is_alpha(char c) noexcept {
        return table_[static_cast<uint8_t>(c)] & char_type_t::Alpha;
    }

    static inline bool is_digit(char c) noexcept {
        return table_[static_cast<uint8_t>(c)] & char_type_t::Digit;
    }

    static inline bool is_identifier_char(char c) noexcept {
        return table_[static_cast<uint8_t>(c)] & char_type_t::Ident;
    }
};