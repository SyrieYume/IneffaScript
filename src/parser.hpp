#pragma once
#include <algorithm>
#include <optional>
#include <string_view>
#include <cstring>
#include "ast_parser.hpp"
#include "bytecode.hpp"

namespace ineffa::script {
struct alignas(uintptr_t) type_info_t {
    enum type_kind_t : uint8_t {
        int_type,
        uint_type,
        float_type,
        string_type,
        array_type,
        function_type,
        void_type,
    };

    type_kind_t type_kind;
    uint16_t size;
    uint16_t info_data_size;

    bool is_integer() const noexcept {
        return type_kind == int_type || type_kind == uint_type;
    }

    auto base_type() noexcept -> type_info_t* {
        return *(type_info_t**)(this + 1);
    }

    auto function_return_type() noexcept -> type_info_t* {
        return *(type_info_t**)(this + 1);
    }

    auto function_params() noexcept -> std::span<type_info_t*> {
        return std::span((type_info_t**)(this + 1) + 1, info_data_size / sizeof(type_info_t*) - 1);
    }
};


struct variable_info_t {
    enum flag_t : uint8_t {
        is_imm_i8 = 0x1,
        is_tmp_reg = 0x2
    };

    type_info_t* type;
    uint8_t reg_id;
    flag_t flags = is_tmp_reg;
    int8_t imm = 0;
};


struct scope_t {
    enum flag_t : uint8_t {
        scope_flag_block = 1 << 0,
        scope_flag_loop =  1 << 1,
        scope_flag_function = 1 << 2,
        scope_flag_global = 1 << 3,

        function_scope = scope_flag_block | scope_flag_function,
        global_scope = function_scope | scope_flag_global,
        block_scope = scope_flag_block,
        loop_scope = scope_flag_block | scope_flag_loop
    };

    std::map<std::string_view, variable_info_t> variables = {};
    std::vector<std::pair<uint32_t, bool>>* jmp_patches = nullptr;
    uint32_t stack_top;
    flag_t flag;
};


class parser_context_t {
private:
    struct span_comparator_t {
        using is_transparent = void;
        bool operator()(std::span<const uintptr_t> lhs, std::span<const uintptr_t> rhs) const {
            return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
        }
    };

public:
    struct function_info_t {
        type_info_t* type;
        uint32_t code_start, code_end;
    };

    std::map<std::string_view, std::unique_ptr<type_info_t>, std::less<>> type_table;
    std::map<std::span<uintptr_t>, std::unique_ptr<uintptr_t[]>, span_comparator_t> function_type_table;
    std::map<std::string_view, uint32_t, std::less<>> functions_map;
    std::vector<function_info_t> functions;
    std::vector<scope_t> scopes;

public:
    parser_context_t() {
        static constexpr std::array<std::pair<std::string_view, type_info_t>, 12> builtin_types = {{
            { "i64",    { type_info_t::int_type, 8, 0 } },
            { "u64",    { type_info_t::uint_type, 8, 0 } },
            { "i32",    { type_info_t::int_type, 4, 0 } },
            { "u32",    { type_info_t::uint_type, 4, 0 } },
            { "i16",    { type_info_t::int_type, 2, 0 } },
            { "u16",    { type_info_t::uint_type, 2, 0 } },
            { "i8",     { type_info_t::int_type, 1, 0 } },
            { "u8",     { type_info_t::uint_type, 1, 0 } },
            { "f64",    { type_info_t::float_type, 8, 0 } },
            { "bool",   { type_info_t::uint_type, 1, 0 } },
            { "string", { type_info_t::string_type, 8, 0 } },
            { "void",   { type_info_t::void_type, 8, 0 } },
        }};

        for (auto [type_name, type_info] : builtin_types)
            type_table.emplace(type_name, new type_info_t(type_info));

        enter_scope(scope_t::global_scope);
    }

    void enter_scope(scope_t::flag_t scope_type) {
        scopes.push_back({ .stack_top = scope_type & scope_t::scope_flag_function ? 0 : scopes.back().stack_top, .flag = scope_type });
    }

    void exit_scope() {
        scopes.pop_back();
    }

    auto continue_or_break_loop(uint32_t current_code_pos, bool continue_or_break) {
        for (auto& scope : scopes | std::views::reverse) {
            if (scope.flag & scope_t::scope_flag_loop) {
                scope.jmp_patches->push_back({current_code_pos, continue_or_break});
                return;
            }
            if (scope.flag & scope_t::scope_flag_function) [[unlikely]] break;
        }
        throw std::runtime_error("cannot find a loop to break/continue");
    }

    auto get_variable(std::string_view name) -> variable_info_t {
        for (auto& scope : scopes | std::views::reverse) {
            if (auto it = scope.variables.find(name); it != scope.variables.end())
                return it->second;
            if (scope.flag & scope_t::scope_flag_function) [[unlikely]] break;
        }
        throw std::runtime_error(std::format("undeclared variable '{}'", name));
    }
    
    auto get_type(std::string_view type_name) -> type_info_t* {
        if (auto it = type_table.find(type_name); it != type_table.end()) [[likely]]
            return it->second.get();

        if (type_name.ends_with("[]")) [[likely]]
            if (auto it = type_table.find(type_name.substr(0, type_name.length() - 2)); it != type_table.end()) [[likely]] {
                type_info_t* base_type = it->second.get();
                type_info_t* type_info = (type_info_t*)::operator new(sizeof(type_info_t) + sizeof(type_info_t*));
                *type_info = { type_info_t::array_type, 8, sizeof(type_info_t*) };
                *(type_info_t**)(type_info + 1) = base_type;
                type_table.emplace(type_name, type_info);
                return type_info;
            }

        throw std::runtime_error(std::format("unknown type: {}", type_name));
    }

    auto get_function(std::string_view name) -> uint32_t {
        if (auto it = functions_map.find(name); it != functions_map.end()) [[likely]]
            return it->second;
        throw std::runtime_error(std::format("undeclared function '{}'", name));
    }

    auto declare_variable(std::string_view name, type_info_t* type) -> variable_info_t {
        uint8_t reg_id = alloc_reg(type->size);

        scope_t& scope = scopes.back();

        if (scope.variables.find(name) != scope.variables.end()) [[unlikely]]
            throw std::runtime_error(std::format("variable '{}' has already been declared", name));

        variable_info_t var_info = { type, reg_id, variable_info_t::flag_t(0) };
        scope.variables.emplace(name, var_info);
        return var_info;
    }

    auto current_stack_top() -> uint32_t& {
        return scopes.back().stack_top;
    }

    auto alloc_tmp_reg() -> uint8_t {
        return (uint8_t)current_stack_top();
    }

    auto alloc_reg(variable_info_t& var) {
        if (uint32_t stack_top = current_stack_top(); var.reg_id >= stack_top)
            alloc_reg((var.reg_id - stack_top) * 8 + var.type->size);
    }

    auto alloc_reg(uint32_t size) -> uint8_t {
        scope_t& scope = scopes.back();
        
        if (uint32_t reg_id = scope.stack_top; reg_id <= std::numeric_limits<uint8_t>::max()) [[likely]] {
            scope.stack_top += (size + 7) / 8;
            return reg_id;
        }
        
        throw std::runtime_error("register overflow");
    }

    auto declare_function(std::string_view name, type_info_t* return_type, std::span<type_info_t*> param_types) -> uint32_t {
        if (functions_map.find(name) != functions_map.end()) [[unlikely]]
            throw std::runtime_error(std::format("function '{}' has already been declared", name));

        uint32_t type_info_size =  (sizeof(type_info_t) + (param_types.size() + 1) * sizeof(uintptr_t)) / sizeof(uintptr_t);
        uintptr_t* type_info_data = new uintptr_t[type_info_size];
        std::unique_ptr<uintptr_t[]> type_info(type_info_data);

        *(type_info_t*)type_info_data = { type_info_t::function_type, 8, uint16_t(sizeof(type_info_t*) * (param_types.size() + 1)) };
        *(type_info_t**)(type_info.get() + 1) = return_type;
        std::memcpy((type_info_t**)(type_info.get() + 1) + 1, param_types.data(), sizeof(type_info_t*) * param_types.size());

        type_info_t* type = (type_info_t*)type_info_data;

        if (auto it = function_type_table.find(std::span<uintptr_t>(type_info_data, type_info_size)); it != function_type_table.end())
            type = (type_info_t*)it->second.get();

        else function_type_table.emplace(std::span<uintptr_t>(type_info_data, type_info_size), std::move(type_info));

        uint32_t func_id = functions.size();
        functions_map.emplace(name, func_id);
        functions.push_back({ type, 0, 0 });

        return func_id;
    }
};


class parser_t {
public:
    struct error_t : std::exception {
        std::string message;
        ast_node_t* error_node;

        error_t(ast_node_t* error_node, const std::string_view msg) : message(msg), error_node(error_node)  {};

        const char* what() const noexcept override {
            return message.c_str();
        }
    };

    auto parse(ast_node_t* root_ast_node, const std::span<std::pair<std::string_view, host_func_t>> host_functions) -> executable_t try {
        assert(root_ast_node != nullptr && root_ast_node->is_list());

        context_ = parser_context_t();
        current_bytecodes_ = {};
        result_bytecodes_ = {};
        host_functions_ = host_functions;
        constants_ = {};
        str_constants_map_.clear();
        result_bytecodes_.push_back({jump, 0, 0});

        for (ast_node_t* node : root_ast_node->get_list())
            parse_expr(node);
        
        current_bytecodes_.push_back({halt, 0, 0});

        std::vector<function_info_t> functions;
        for (auto func_info : context_.functions) {
            functions.push_back({
                .start = func_info.code_start, 
                .end = func_info.code_end, 
                .ret_val_size = func_info.type->function_return_type()->size  
            });
        }

        uint32_t main_start = result_bytecodes_.size();
        uint32_t main_size = current_bytecodes_.size();
        functions.push_back({ .start = main_start, .end = main_start + main_size, .ret_val_size = 0 });
        
        result_bytecodes_[0].u.imm = (int16_t)result_bytecodes_.size() - 1;
        result_bytecodes_.append_range(current_bytecodes_);

        for (auto [i, ins] : result_bytecodes_ | std::views::enumerate)
            if (ins.opcode == call)
                ins.u.imm = (int16_t)(context_.functions[ins.u.imm].code_start - i - 1);

        return executable_t {
            .bytecodes = std::move(result_bytecodes_),
            .constants = std::move(constants_),
            .functions = std::move(functions)
        };
    }
    catch(std::exception& e) {
        throw error_t(current_node_, e.what());
    }

private:
    using enum opcode_t;

    static inline std::map<std::string_view, std::pair<std::string_view, std::array<opcode_t, 3>>> infix_syntaxs = { 
        { "=",  { "", { move_64, move_64, move_64 } } },
        { "at", { "", { load_64, load_64, load_64 } } },
        { "==", { "bool", {set_is_equal_64, set_is_equal_64, halt} } },
        { "!=", { "bool", {set_is_not_equal_64, set_is_not_equal_64, halt} } },
        { "<",  { "bool", {set_is_less_than_i64, set_is_less_than_u64, set_is_less_than_f64} } },
        { ">",  { "bool", {set_is_less_than_i64, set_is_less_than_u64, set_is_less_than_f64} } },
        { "<=", { "bool", {set_is_less_equal_i64, set_is_less_equal_u64, halt} } },
        { ">=", { "bool", {set_is_less_equal_i64, set_is_less_equal_u64, halt} } },
        { "+",  { "", {add_64, add_64, add_f64} } },
        { "-",  { "", {sub_64, sub_64, sub_f64} } },
        { "*",  { "", {mul_64, mul_64, mul_f64} } },
        { "/",  { "", {div_i64, div_u64, div_f64} } },
        { "%",  { "", {mod_i64, mod_u64, halt} } },
    };

    struct stack_top_guard_t {
        parser_context_t& context;
        uint32_t stack_top;
        stack_top_guard_t(parser_context_t& context) : context(context), stack_top(context.current_stack_top()) {}
        ~stack_top_guard_t() {
            context.current_stack_top() = stack_top;
        }
    };

    auto parse_let(const std::span<ast_node_t*> node_list) -> variable_info_t {
        std::string_view name = get_atom(node_list, 1);
        std::string_view type_name = get_atom(node_list, 2).substr(1);
        type_info_t* type = context_.get_type(type_name);
        
        if (!get_atom(node_list, 3u, false, "=").empty()) {
            variable_info_t var = { type, context_.alloc_tmp_reg() };
            parse_list_node(node_list.subspan(4), &var);
        }

        return context_.declare_variable(name, type);
    }

    auto parse_assign(const std::span<ast_node_t*> node_list) -> variable_info_t {
        if (!get_atom(node_list[0]->get_list(), 1, false, "at").empty()) {
            stack_top_guard_t stack_top_guard(context_);
            variable_info_t src = parse_list_node(node_list.subspan(2));
            context_.alloc_reg(src);
            return parse_at(node_list[0]->get_list(), false, &src);
        }

        variable_info_t var = parse_expr(node_list[0]);
        return parse_list_node(node_list.subspan(2), &var);
    }

    auto parse_at(const std::span<ast_node_t*> node_list, bool is_load_or_store, variable_info_t* dest_or_src_var = nullptr) -> variable_info_t {
        assert(is_load_or_store || dest_or_src_var);
        stack_top_guard_t stack_top_guard(context_);

        variable_info_t array_var = parse_expr(get_expr(node_list, 0));
        context_.alloc_reg(array_var);
        variable_info_t index_var = parse_expr(get_expr(node_list, 2));
        
        if (array_var.type->type_kind != type_info_t::array_type || !index_var.type->is_integer()) [[unlikely]]
            throw std::runtime_error("syntax 'at' only support array type and interger type");

        type_info_t* base_type = array_var.type->base_type();
        variable_info_t dest_or_src = dest_or_src_var ? *dest_or_src_var : variable_info_t { base_type, context_.alloc_tmp_reg() };

        const opcode_t op = [&] {
            if (is_load_or_store) {
                bool is_signed = (base_type->type_kind == type_info_t::int_type);
                switch (base_type->size) {
                    case 8: return load_64;
                    case 4: return is_signed ? load_i32 : load_u32;
                    case 2: return is_signed ? load_i16 : load_u16;
                    case 1: return is_signed ? load_i8 : load_u8;
                    default: throw std::runtime_error("unsupport type size"); 
                }
            }
            else switch (base_type->size) {
                case 8: return store_64;
                case 4: return store_32;
                case 2: return store_16;
                case 1: return store_8;
                default: throw std::runtime_error("unsupport type size");
            }
        }();

        emit({op, dest_or_src.reg_id, array_var.reg_id, index_var.reg_id});
        return dest_or_src;
    }

    auto parse_block(const std::span<ast_node_t*> node_list, variable_info_t* dest_var = nullptr) -> variable_info_t {
        std::optional<variable_info_t> ret_var;
        
        for (uint32_t i = 0; i < node_list.size(); i++) {
            if (i == node_list.size() - 1) [[unlikely]]
                ret_var = parse_expr(node_list[i], dest_var);
            else parse_expr(node_list[i]);
        }
        
        return ret_var.value_or(variable_info_t { context_.get_type("void"), context_.alloc_tmp_reg() });
    }

    auto parse_if(const std::span<ast_node_t*> node_list, variable_info_t* dest_var = nullptr) -> variable_info_t {
        context_.enter_scope(scope_t::block_scope);

        uint32_t condition_jump_code_pos = [&] {
            ast_node_t* condition_expr = get_expr(node_list, 1);
            std::span<ast_node_t*> cond_list = condition_expr->get_list();
            std::string_view infix_atom = get_atom(cond_list, 1u, false);

            if (infix_atom == "==" && !get_atom(cond_list, 2, false, "0").empty())
                return emit({jump_if_true, parse_expr(get_expr(cond_list, 0u)).reg_id, 0});

            if (infix_atom == "!=" && !get_atom(cond_list, 2, false, "0").empty())
                return emit({jump_if_false, parse_expr(get_expr(cond_list, 0)).reg_id, 0});

            variable_info_t condition_var = parse_expr(condition_expr);
            
            if (condition_var.flags & variable_info_t::is_tmp_reg) {
                instruction_t& last_ins = current_bytecodes_.back();

                if (last_ins.opcode == set_is_less_than_i64) {
                    last_ins = { jump_if_less_equal_i64, last_ins.r.rs2, last_ins.r.rs1 };
                    return uint32_t(current_bytecodes_.size() - 1);
                }

                if (last_ins.opcode == set_is_less_equal_i64) {
                    last_ins = { jump_if_less_than_i64, last_ins.r.rs2, last_ins.r.rs1 };
                    return uint32_t(current_bytecodes_.size() - 1);
                }
            }
            return emit({jump_if_false, condition_var.reg_id, 0});
        }();

        auto else_it = std::find_if(node_list.begin() + 2, node_list.end(), [](ast_node_t* node) { return node->get_atom() == "else"; });

        bool has_else_branch = else_it != node_list.end();
        uint32_t else_branch_code_pos;
        uint32_t then_branch_jump_to_end_code_pos;

        variable_info_t then_branch_ret_var = parse_block(std::span<ast_node_t*>(node_list.begin() + 2, else_it), dest_var);

        if (has_else_branch) {
            context_.exit_scope();
            context_.enter_scope(scope_t::block_scope);
            then_branch_jump_to_end_code_pos = emit({jump, 0, 0});
            else_branch_code_pos = current_bytecodes_.size();
            [[maybe_unused]] variable_info_t else_branch_ret_var = parse_block(std::span<ast_node_t*>(else_it + 1, node_list.end()), dest_var);
        }

        context_.exit_scope();

        emit_jmp(jump_if_false, has_else_branch ? else_branch_code_pos : current_bytecodes_.size(), condition_jump_code_pos);
        
        if (has_else_branch)
            emit_jmp(jump, current_bytecodes_.size(), then_branch_jump_to_end_code_pos);
        
        return then_branch_ret_var;
    }

    auto parse_loop(const std::span<ast_node_t*> node_list, variable_info_t* dest_var = nullptr) -> variable_info_t {
        std::vector<std::pair<uint32_t, bool>> jmp_patches;

        context_.enter_scope(scope_t::loop_scope);
        uint32_t loop_start = current_bytecodes_.size();
        context_.scopes.back().jmp_patches = &jmp_patches;
        
        for (ast_node_t* node : node_list.subspan(1))
            parse_expr(node);

        emit_jmp(jump, loop_start);
        context_.exit_scope();

        uint32_t loop_end = current_bytecodes_.size();
        for (auto [code_pos, continue_or_break] : jmp_patches)
            emit_jmp(jump, continue_or_break ? loop_start : loop_end, code_pos);

        return variable_info_t { context_.get_type("void"), context_.alloc_tmp_reg() };
    }

    auto parse_for(const std::span<ast_node_t*> node_list, variable_info_t* dest_var = nullptr) -> variable_info_t {
        std::vector<std::pair<uint32_t, bool>> jmp_patches;

        context_.enter_scope(scope_t::loop_scope);
        context_.scopes.back().jmp_patches = &jmp_patches;

        auto to_it = std::find_if(node_list.begin() + 2, node_list.end(), [=](ast_node_t* node) { return node->is_atom() && node->get_atom() == "to"; });
        if (to_it == node_list.end()) [[unlikely]]
            throw std::runtime_error("expect 'to' in for loop expr");

        uint32_t to_pos = std::distance(node_list.begin(), to_it);
        variable_info_t var_i = parse_list_node(node_list.subspan(1, to_pos - 1));
        context_.alloc_reg(var_i);
        variable_info_t var_limit = parse_expr(get_expr(node_list, to_pos + 1));
        context_.alloc_reg(var_limit);

        emit({set_is_less_than_i64, context_.alloc_tmp_reg(), var_i.reg_id, var_limit.reg_id  });
        context_.continue_or_break_loop(emit({jump_if_false, context_.alloc_tmp_reg(), 0}), false);
        
        uint32_t loop_start = current_bytecodes_.size();
        for (ast_node_t* node : node_list.subspan(to_pos + 2))
            parse_expr(node);

        emit_jmp(loop_inc_check_jump, loop_start, -1, var_i.reg_id, var_limit.reg_id);
        context_.exit_scope();

        uint32_t continue_target = current_bytecodes_.size() - 1;
        uint32_t loop_end = current_bytecodes_.size();
        for (auto [code_pos, continue_or_break] : jmp_patches)
            emit_jmp(jump, continue_or_break ? continue_target : loop_end, code_pos);

        return variable_info_t { context_.get_type("void"), context_.alloc_tmp_reg() };
    }

    auto parse_call(const std::span<ast_node_t*> node_list, bool is_host) -> variable_info_t {
        std::string_view func_name = get_atom(node_list, 1);
        uint8_t stack_top = context_.current_stack_top();
        stack_top_guard_t stack_top_guard(context_);

        for (ast_node_t* node : node_list.subspan(2)) {
            variable_info_t var = { nullptr, context_.alloc_tmp_reg() };
            var = parse_expr(node, &var);
            context_.alloc_reg(8);
        }

        if (is_host) {
            type_info_t* function_return_type = context_.get_type("void");
            auto it = std::ranges::find_if(host_functions_, [=](auto& v) { return v.first == func_name; });

            if (it == host_functions_.end()) [[unlikely]]
                throw std::runtime_error(std::format("unknown host function: '{}'", func_name));

            emit({ call_host, stack_top, (int16_t)(uint16_t)std::distance(host_functions_.begin(), it) });
            return { function_return_type, stack_top };
        }
        else {
            uint32_t func_id = context_.get_function(func_name);
            type_info_t* function_return_type = context_.functions[func_id].type->function_return_type();
            emit({ call, stack_top, (int16_t)func_id });
            return { function_return_type, stack_top };
        }
    }

    auto parse_defun(const std::span<ast_node_t*> node_list, variable_info_t* dest_var = nullptr) -> variable_info_t {
        std::vector<instruction_t> last_bytecodes_part = std::move(current_bytecodes_);
        current_bytecodes_ = std::vector<instruction_t>();
        context_.enter_scope(scope_t::function_scope);

        std::string_view func_name = get_atom(node_list, 1);
        ast_node_t* args_expr = get_expr(node_list, 2);

        if (args_expr->is_atom()) [[unlikely]]
            throw std::runtime_error("expect list node");

        std::span<ast_node_t*> args = args_expr->get_list();
        std::vector<type_info_t*> param_types;

        if (get_atom(args, 0, false, ":void").empty())
            for (uint32_t pos = 0; pos < args.size(); pos += 2) {
                std::string_view arg_name = get_atom(args, pos);
                std::string_view type_name = get_atom(args, pos + 1).substr(1);
                type_info_t* type = context_.get_type(type_name);

                context_.declare_variable(arg_name, type);
                param_types.push_back(type);
            }

        type_info_t* return_type = context_.get_type(get_atom(node_list, 3).substr(1));

        uint32_t func_id = context_.declare_function(func_name, return_type, param_types);

        for (ast_node_t* node : node_list.subspan(4))
            parse_expr(node);

        if (current_bytecodes_.back().opcode != ret)
            emit({ret, 0, 0});

        context_.exit_scope();

        context_.functions[func_id].code_start = result_bytecodes_.size();
        result_bytecodes_.append_range(current_bytecodes_);
        context_.functions[func_id].code_end = result_bytecodes_.size();

        current_bytecodes_ = std::move(last_bytecodes_part);

        return variable_info_t { context_.functions[func_id].type, context_.alloc_tmp_reg() };
    }

    auto parse_return(const std::span<ast_node_t*> node_list) -> variable_info_t {
        variable_info_t var = { nullptr, 0 };
        parse_expr(get_expr(node_list, 1), &var);
        emit({ret, 0, 0});
        return { context_.get_type("void"), context_.alloc_tmp_reg() };
    }

    auto parse_infix_expression(decltype(infix_syntaxs)::iterator it, std::span<ast_node_t*> node_list, variable_info_t* dest_var) -> variable_info_t {
        std::string_view syntax_name = it->first;
        std::string_view return_type_name = it->second.first;
        std::array<opcode_t, 3> opcodes = it->second.second;

        if (syntax_name == "=")
            return parse_assign(node_list);

        if (syntax_name == "at")
            return parse_at(node_list, true);

        stack_top_guard_t stack_top_guard(context_);

        variable_info_t var1 = parse_expr(get_expr(node_list, 0));
        context_.alloc_reg(var1);

        bool is_handle_imm_i8 = var1.type->is_integer() && (syntax_name == "+" || syntax_name == "-");
        variable_info_t var2 = parse_expr(get_expr(node_list, 2), nullptr, is_handle_imm_i8);

        bool types_ok = (var1.type == var2.type) ||
            ((syntax_name == "+" || syntax_name == "-" || syntax_name == "*") && var1.type->is_integer() && var2.type->is_integer());

        if (!types_ok) [[unlikely]]
            throw std::runtime_error(std::format("unsupport syntax '{}' between diffrent types", syntax_name));

        type_info_t* return_type = return_type_name.empty() ? var1.type : context_.get_type(return_type_name);

        variable_info_t dest = 
            dest_var ? *dest_var :
            var1.flags & variable_info_t::is_tmp_reg ? var1 :
            var2.flags & variable_info_t::is_tmp_reg ? var2 :
            variable_info_t { return_type, context_.alloc_tmp_reg() };


        if (syntax_name == ">" || syntax_name == ">=")
            std::swap(var1, var2);

        if (dest.flags & variable_info_t::is_tmp_reg)
            dest.type = return_type;

        if (var2.flags & variable_info_t::is_imm_i8) {
            if (syntax_name == "-")
                var2.imm = -var2.imm;

            emit({add_imm_i8, dest.reg_id, var1.reg_id, var2.imm});
        }

        else {
            opcode_t opcode = var1.type->type_kind < 3 ? opcodes[var1.type->type_kind] : halt;
            if (opcode == halt) [[unlikely]]
                throw std::runtime_error("unsupport syntax '{}' for this type");
            
            emit({opcode, dest.reg_id, var1.reg_id, var2.reg_id});
        }

        return dest;
    }

    auto parse_list_node(std::span<ast_node_t*> node_list, variable_info_t* dest_var = nullptr) -> variable_info_t {
        auto hash_str = [](std::string_view sv) constexpr -> uint64_t {
            uint64_t hash = 14695981039346656037ull;
            constexpr uint64_t prime = 1099511628211ull;
            for (char c : sv) hash = (hash ^ uint64_t(c)) * prime;
            return hash;
        };

        variable_info_t var = [&] {
            if (node_list.size() <= 1)
                return parse_expr(get_expr(node_list, 0), dest_var);

            if (auto it = infix_syntaxs.find(get_atom(node_list, 1, false)); it != infix_syntaxs.end())
                return parse_infix_expression(it, node_list, dest_var);

            switch (auto syntax_name = get_atom(node_list, 0); hash_str(syntax_name)) {
                case hash_str("let")      : return parse_let(node_list);
                case hash_str("if")       : return parse_if(node_list, dest_var);
                case hash_str("loop")     : return parse_loop(node_list, dest_var);
                case hash_str("for")      : return parse_for(node_list, dest_var);
                case hash_str("callhost") : return parse_call(node_list, true);
                case hash_str("call")     : return parse_call(node_list, false);
                case hash_str("defun")    : return parse_defun(node_list);
                case hash_str("return")   : return parse_return(node_list); 
                default: throw std::runtime_error(std::format("invalid syntax: {}", syntax_name));
            }
        }();

        if (dest_var && var.reg_id != dest_var->reg_id)
            emit({ move_64, dest_var->reg_id, var.reg_id, int8_t(0) });

        return dest_var ? *dest_var : var;
    }

    auto parse_atom_node(std::string_view atom, variable_info_t* dest_var = nullptr, bool return_imm_i8 = false) -> variable_info_t {
        uint8_t reg_id = dest_var ? dest_var->reg_id : context_.alloc_tmp_reg();
        variable_info_t::flag_t var_flag = dest_var ? dest_var->flags : variable_info_t::is_tmp_reg;

        if (char_utils::is_digit(atom[0]) || atom[0] == '-') {
            bool is_float = atom.contains('.');

            int64_t value;
            auto result = is_float ? 
                std::from_chars(atom.data(), atom.data() + atom.length(), *(double*)&value) :
                std::from_chars(atom.data(), atom.data() + atom.length(), value);
            
            if (result.ec == std::errc::invalid_argument) [[unlikely]]
                throw std::runtime_error("expected integer literal");

            else if (result.ec == std::errc::result_out_of_range) [[unlikely]]
                throw std::runtime_error("integer literal out of range");

            type_info_t* value_type = context_.get_type(is_float ? "f64" : atom.ends_with('u') ? "u64" : "i64");

            if (is_float || (value < std::numeric_limits<int16_t>::min() || value > std::numeric_limits<int16_t>::max())) {
                if (constants_.size() / 8 > std::numeric_limits<uint16_t>::max()) [[unlikely]]
                    throw std::runtime_error("constants pool overflow");
        
                emit({load_constant_64, reg_id, int16_t(constants_.size() / 8)});

                constants_.insert(constants_.end(), (uint8_t*)(&value), (uint8_t*)(&value) + 8);
                return variable_info_t { value_type, reg_id, var_flag };
            }
            
            else if (return_imm_i8 && std::abs(value) <= std::numeric_limits<int8_t>::max())
                return variable_info_t { value_type, reg_id, variable_info_t::is_imm_i8, int8_t(value) }; 

            else {
                emit({load_imm_i16, reg_id, int16_t(value)});
                return variable_info_t { value_type, reg_id, var_flag };
            }
        }

        else if (atom[0] == '"') {
            emit({load_constant_str, reg_id, (int16_t)parse_string_literal(atom)});
            return variable_info_t { context_.get_type("string"), reg_id, var_flag };
        }
        
        else if (char_utils::is_identifier_char(atom[0])) {
            if (atom == "true" || atom == "false") {
                emit({load_imm_i16, reg_id, int16_t(atom.length() == 4)});
                return variable_info_t { context_.get_type("bool"), reg_id, var_flag };
            }
            
            if (atom == "break") {
                context_.continue_or_break_loop(current_bytecodes_.size(), false);
                emit({jump, 0, 0});
                return variable_info_t { context_.get_type("void"), reg_id, var_flag };
            }

            if (atom == "continue") {
                context_.continue_or_break_loop(current_bytecodes_.size(), false);
                emit({jump, 0, 0});
                return variable_info_t { context_.get_type("void"), reg_id, var_flag };
            }

            if (atom == "return") {
                emit({ret, 0, 0});
                return variable_info_t { context_.get_type("void"), reg_id, var_flag };
            }
            
            variable_info_t var = context_.get_variable(atom);
            if (dest_var && var.reg_id != reg_id)
                emit({ move_64, reg_id, var.reg_id, int8_t(0) });
            return var;
        }

        throw std::runtime_error(std::format("invalid atom: '{}'", atom));
    }

    auto parse_expr(ast_node_t* node, variable_info_t* dest_var = nullptr, bool return_imm_i8 = false) -> variable_info_t {
        return node->is_atom() ?
            parse_atom_node(node->get_atom(), dest_var, return_imm_i8) :
            parse_list_node(node->get_list(), dest_var);
    }

    auto get_expr(std::span<ast_node_t*> node_list, uint32_t pos) -> ast_node_t* {
        if (pos >= node_list.size()) [[unlikely]]
            throw std::runtime_error(std::format("expect expression at pos {}", pos));
        return current_node_ = node_list[pos];
    }

    auto get_atom(std::span<ast_node_t*> node_list, uint32_t pos, bool is_required = true, std::string_view expected = "") -> std::string_view {
        if (pos >= node_list.size() || !node_list[pos]->is_atom()) [[unlikely]] {
            if (is_required)
                throw std::runtime_error(std::format("expect atom at pos {}", pos));
            return "";
        }

        current_node_ = node_list[pos];

        std::string_view atom = node_list[pos]->get_atom();

        if (expected.empty() || atom == expected)
            return atom;

        if (is_required) [[unlikely]]
            throw std::runtime_error(std::format("expect {} at pos {}", expected, pos));

        return "";
    }

    auto parse_string_literal(const std::string_view source) -> uint16_t {
        uint32_t constant_start = constants_.size();

        for (uint32_t i = 1; i < source.size(); i++) {
            char c = source[i];
            if (c == '"') [[unlikely]]
                break;

            if (c == '\\') {
                c = source[++i];
                if (c == 'n') c = '\n';
                if (c == 'r') c = '\r';
                if (c == 't') c = '\t';
            }
            
            constants_.push_back((uint8_t)c);
        }

        constants_.push_back(0);

        std::string_view sv((char*)(constants_.data() + constant_start), constants_.size() - constant_start - 1);
        if (auto it = str_constants_map_.find(sv); it != str_constants_map_.end()) {
            constants_.resize(constant_start); 
            return it->second;
        }

        if (constant_start / 8 > std::numeric_limits<uint16_t>::max()) [[unlikely]]
            throw std::runtime_error("constants pool overflow");

        str_constants_map_.emplace(sv, constant_start / 8);

        uint32_t current_size = constants_.size();
        uint32_t aligned_size = (current_size + 7) & ~uint32_t(7);

        if (aligned_size > current_size)
            constants_.resize(aligned_size, 0);
        
        return uint16_t(constant_start / 8);
    }

    auto emit(instruction_t ins) -> uint32_t {
        current_bytecodes_.push_back(ins);
        return current_bytecodes_.size() - 1;
    }

    auto emit_jmp(opcode_t opcode, int64_t target, int64_t code_pos = -1, int32_t rs1 = -1, int32_t rs2 = -1) -> uint32_t {
        int64_t from_code_pos = code_pos >= 0 ? code_pos : current_bytecodes_.size();
        if (code_pos < 0)
            current_bytecodes_.push_back({opcode, 0, 0});

        int64_t offset = target - from_code_pos - 1;
        
        instruction_t& from_ins = current_bytecodes_[from_code_pos];
        switch (from_ins.opcode) {
            case jump:
            case jump_if_false:
            case jump_if_true:
                if (offset < std::numeric_limits<int16_t>::min() || offset > std::numeric_limits<int16_t>::max()) [[unlikely]]
                    throw std::runtime_error("this function/loop/if is too far/long to call/jump");
                from_ins.u.imm = (int16_t)offset;
                if (rs1 >= 0) from_ins.rd = (uint8_t)rs1;
                break;

            case loop_inc_check_jump:
                if (-offset > std::numeric_limits<uint8_t>::max()) [[unlikely]]
                    throw std::runtime_error("this loop is too long to jump");
                from_ins.i.imm = (uint8_t)-offset;
                if (rs1 >= 0) from_ins.rd = (uint8_t)rs1;
                if (rs2 >= 0) from_ins.i.rs1 = (uint8_t)rs2;
                break;
            
            case jump_if_less_than_i64:
            case jump_if_less_equal_i64:
                if (offset < std::numeric_limits<int8_t>::min() || offset > std::numeric_limits<int8_t>::max()) [[unlikely]]
                    throw std::runtime_error("this function/loop/if is too far/long to call/jump");
                from_ins.i.imm = (int8_t)offset;
                if (rs1 >= 0) from_ins.rd = (uint8_t)rs1;
                if (rs2 >= 0) from_ins.i.rs1 = (uint8_t)rs2;
                break;
            default: break;
        };
            
        return from_code_pos;
    }

private:
    parser_context_t context_;
    std::vector<instruction_t> current_bytecodes_;
    ast_node_t* current_node_;
    std::span<std::pair<std::string_view, host_func_t>> host_functions_;
    std::vector<uint8_t> constants_;
    std::map<std::string, uint32_t, std::less<>> str_constants_map_;
    std::vector<instruction_t> result_bytecodes_;
};

}

[[maybe_unused]] static inline auto parser_exmaple(std::string_view source_code) -> ineffa::script::executable_t {
    using namespace ineffa::script;

    std::vector<std::pair<std::string_view, host_func_t>> host_functions = {};

    std::pmr::monotonic_buffer_resource memory_pool = std::pmr::monotonic_buffer_resource(16 * 1024, std::pmr::new_delete_resource());
    std::pmr::polymorphic_allocator<std::byte> memory_allocator(&memory_pool);
    
    ast_node_t* root_node = ast_parser_t(memory_allocator).parse(source_code);
    executable_t exe = parser_t().parse(root_node, host_functions);

    return exe;
}