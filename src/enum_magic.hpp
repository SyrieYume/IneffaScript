#pragma once
#include <ranges>
#include <string_view>
#include <array>
#include <map>

namespace ineffa::compile_time_utilitys {
    static constexpr auto string_trim(std::string_view text) -> std::string_view {
        auto is_space = [](char c) { return c == ' ' || c == '\n' || c == '\t'; };
        auto start = std::ranges::find_if_not(text, is_space);
        auto end = std::ranges::find_if_not(text | std::views::reverse, is_space).base();
        return (start < end) ? std::string_view(start, end) : "";
    }

    template <size_t N>
    struct range_to_array {
        template <std::ranges::input_range R>
        friend consteval auto operator|(R&& r, range_to_array) {
            using T = std::ranges::range_value_t<R>;
            std::array<T, N> result;
            auto it = std::ranges::begin(r);
            for (size_t i = 0; i < N && it != std::ranges::end(r); ++i, ++it)
                result[i] = static_cast<T>(*it);
            return result;
        }
    };
    
    template <size_t N>
    static constexpr auto string_split(std::string_view text, char separator) {
        return text 
            | std::views::split(separator) 
            | std::views::transform([](auto&& r) -> std::string_view { return string_trim(std::string_view(&*r.begin(), std::ranges::distance(r))); })
            | range_to_array<N>();
    }

};

#define generate_enum_class_and_map(enum_base_type, enum_class_name, enum_to_string_array, string_to_enum_map, ...) \
    enum class enum_class_name : enum_base_type { \
        __VA_ARGS__ \
    }; \
    static auto enum_to_string_array = []() { \
        constexpr std::string_view raw = #__VA_ARGS__; \
        constexpr auto count = std::ranges::distance(raw | std::views::split(',')); \
        return ineffa::compile_time_utilitys::string_split<count>(raw, ','); \
    }(); \
    static auto string_to_enum_map = []() -> std::map<std::string_view, enum_class_name> { \
        std::map<std::string_view, enum_class_name> result; \
        for (auto [i, name] : enum_to_string_array | std::views::enumerate) \
            result.emplace(name, static_cast<enum_class_name>(i)); \
        return result; \
    }();
