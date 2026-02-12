#pragma once
#include <memory_resource>
#include <format>
#include <vector>
#include <span>
#include <cstring>
#include "char_utils.hpp"

namespace ineffa::script {

struct alignas(uintptr_t) ast_node_t {
    enum node_type_t : uint8_t { type_atom, type_list };
    node_type_t type;
    uint16_t line;
    uint32_t data_size;

    bool is_atom() const noexcept { return type == type_atom; }

    bool is_list() const noexcept { return type == type_list; } 

    auto get_atom() -> std::string_view {
        char* str = *(char**)(this + 1);
        return is_atom() ? std::string_view(str, str + data_size) : "";
    }

    auto get_list() -> std::span<ast_node_t*> {
        return is_list() ? std::span((ast_node_t**)(this + 1), data_size / sizeof(ast_node_t*)) : std::span<ast_node_t*>((ast_node_t**)0, (ast_node_t**)0);
    }
};

class ast_parser_t {
public:
    ast_parser_t(std::pmr::polymorphic_allocator<std::byte> allocator) : allocator_(allocator) {}

    auto parse(std::string_view source_code) -> ast_node_t* {
        current_ = source_code.data();
        end_ = source_code.data() + source_code.size();
        line_ = 1;
        
        std::vector<ast_node_t*> root_exprs;
        
        while (true) {
            skip();

            if (current_ >= end_) [[unlikely]]
                break;

            root_exprs.push_back(parse_expr());
        }

        return alloc_node(ast_node_t::type_list, root_exprs.size() * sizeof(void*), root_exprs.data());
    }

private:
    [[noreturn]] void error(std::string_view msg) {
        throw std::runtime_error(std::format("[ast_parser_error] [line: {}] {}", line_, msg));
    }

    void skip() noexcept {
        while (current_ < end_) {
            if (match('\n')) [[unlikely]]
                line_++;

            else if (char_utils::is_space(*current_)) [[likely]]
                current_++;

            else if (match(';'))
                for (; current_ < end_ && *current_ != '\n'; current_++);

            else break;
        }
    }

    bool match(char expected) noexcept {
        return *current_ == expected ? (++current_, true) : false;
    }

    ast_node_t* alloc_node(ast_node_t::node_type_t type, uint32_t data_len, const void* data) {
        size_t bytes = sizeof(ast_node_t) + (type == ast_node_t::type_atom ? sizeof(const char*) : data_len);
        auto* node = (ast_node_t*)allocator_.allocate_bytes(bytes, alignof(ast_node_t));
        *node = ast_node_t(type, line_, data_len);

        return type == ast_node_t::type_list ?
            (std::memcpy(node + 1, data, data_len), node) :
            (*(const char**)(node + 1) = (const char*)data, node);
    }

    ast_node_t* parse_expr() {
        if (skip(); current_ >= end_) [[unlikely]]
            error("unexpected end of file");

        if (match(')')) [[unlikely]]
            error("unexpected ')'");

        if (match('(')) {
            std::vector<ast_node_t*> list;

            while (true) {
                if (skip(); current_ >= end_) [[unlikely]]
                    error("unexpected enf of file, missing ')'");

                if (match(')')) [[unlikely]] 
                    break;

                list.push_back(parse_expr());
            }

            return alloc_node(ast_node_t::type_list, list.size() * sizeof(uintptr_t), list.data());
        }

        const char* start = current_;

        if (match('"')) {
            for (;;current_++) {
                if (current_ >= end_) [[unlikely]]
                    error("unterminated string literal");

                if (match('"'))
                    break;

                match('\\');
            }
        }

        else for (char c; current_ < end_ && (c = *current_, !char_utils::is_space(c) && c != ')' && c != '('); current_++);

        return alloc_node(ast_node_t::type_atom, (uint32_t)(current_ - start), start);
    }

    std::pmr::polymorphic_allocator<std::byte> allocator_;
    const char* current_;
    const char* end_;
    uint32_t line_;
};

}

[[maybe_unused]] static inline void ast_parser_exmaple(std::string_view source_code) {
    using namespace ineffa::script;
    std::pmr::monotonic_buffer_resource memory_pool = std::pmr::monotonic_buffer_resource(16 * 1024, std::pmr::new_delete_resource());
    std::pmr::polymorphic_allocator<std::byte> memory_allocator(&memory_pool);
    [[maybe_unused]] ast_node_t* root_node = ast_parser_t(memory_allocator).parse(source_code);
}