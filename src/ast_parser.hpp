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

    auto parse(std::string_view source_code) const -> ast_node_t* try {
        source_ = source_code;
        current_ = 0;
        
        return parse_list_node();
    }
    catch(const std::exception& e) {
        std::string_view parsed_str = source_.substr(0, current_);
        int line = std::count(parsed_str.begin(), parsed_str.end(), '\n') + 1;
        int col = current_ - parsed_str.rfind('\n');
        throw std::runtime_error(std::format("[ast_parser_error][{},{}] {}", line, col, e.what()));
    }

private:
    auto is_at_end() const noexcept -> bool {
        return current_ >= source_.length();
    }

    auto peek() const noexcept -> char {
        if (is_at_end()) [[unlikely]]
            return '\0';
        return source_[current_];
    }

    bool match(char expected) const {
        if (peek() == expected) {
            current_++;
            return true;
        }
        return false;
    }

    template <typename Func>
    requires requires(Func func, char c) { { func(c) } -> std::convertible_to<bool>; }
    auto match_while(Func&& condition) const noexcept -> std::string_view {
        const uint32_t start = current_;
        uint32_t current = current_;
        const size_t len = source_.length();
        const char* data = source_.data();

        while (current < len && condition(data[current]))
            ++current;

        current_ = current;
        return std::string_view(data + start, current - start);
    }


    auto parse_list_node() const -> ast_node_t* {
        std::vector<ast_node_t*> buffer;

        while (true) {
            match_while(char_utils::is_space);

            if (is_at_end() || match(')'))
                break;
            
            if (match(';')) {
                match_while([](char c) { return c != '\n'; });
                continue;
            }
            
            if (match('(')) {
                buffer.push_back(parse_list_node());
                continue;
            }
                
            if (match('"')) {
                std::string_view text = match_while([](char c) { return c != '"'; });
                if (is_at_end() || !match('"')) [[unlikely]]
                    throw std::runtime_error("unterminated string literal");
                buffer.push_back(make_atom_node(std::string_view(text.data() - 1, text.data() + text.length() + 1)));
            }

            else buffer.push_back(make_atom_node(match_while([](char c) { return !char_utils::is_space(c) && c != ')' && c != '(' && c != '"'; })));
        }
        
        return make_list_node(buffer);
    }

    auto make_atom_node(std::string_view text) const -> ast_node_t* {
        void* node_memory = allocator_.allocate_bytes(sizeof(ast_node_t) + sizeof(char*), alignof(ast_node_t));
        ast_node_t* node = (ast_node_t*)std::assume_aligned<8>(node_memory);
        *node = { ast_node_t::type_atom, (uint32_t)text.length() };
        *(const char**)(node + 1) = text.data();
        return node;
    }

    auto make_list_node(std::span<ast_node_t*> list) const -> ast_node_t* {
        void* node_memory = allocator_.allocate_bytes(sizeof(ast_node_t) + sizeof(ast_node_t*) * list.size(), alignof(ast_node_t));
        ast_node_t* node = (ast_node_t*)std::assume_aligned<8>(node_memory);
        *node = { ast_node_t::type_list, uint32_t(sizeof(ast_node_t*) * list.size()) };
        std::memcpy(node + 1, list.data(), sizeof(ast_node_t*) * list.size());
        return node;
    }

private:
    mutable std::pmr::polymorphic_allocator<std::byte> allocator_;
    mutable std::string_view source_;
    mutable uint32_t current_;
};

}

[[maybe_unused]] static inline void ast_parser_exmaple(std::string_view source_code) {
    using namespace ineffa::script;
    std::pmr::monotonic_buffer_resource memory_pool = std::pmr::monotonic_buffer_resource(16 * 1024, std::pmr::new_delete_resource());
    std::pmr::polymorphic_allocator<std::byte> memory_allocator(&memory_pool);
    [[maybe_unused]] ast_node_t* root_node = ast_parser_t(memory_allocator).parse(source_code);
}