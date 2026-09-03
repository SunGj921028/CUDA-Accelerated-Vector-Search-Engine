#pragma once

#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vector_search_test {

struct TestCase {
    const char* name;
    void (*function)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

class Registrar {
public:
    Registrar(const char* name, void (*function)()) {
        registry().push_back({name, function});
    }
};

[[noreturn]] inline void fail(
    const std::string& message,
    const char* file,
    int line) {
    std::ostringstream output;
    output << file << ':' << line << ": " << message;
    throw std::runtime_error(output.str());
}

inline void require(
    bool condition,
    const char* expression,
    const char* file,
    int line) {
    if (!condition) {
        fail(std::string("requirement failed: ") + expression, file, line);
    }
}

template <typename Left, typename Right>
void require_equal(
    const Left& left,
    const Right& right,
    const char* expression,
    const char* file,
    int line) {
    if (!(left == right)) {
        std::ostringstream message;
        message << "expected " << expression;
        fail(message.str(), file, line);
    }
}

inline void require_near(
    float actual,
    float expected,
    float tolerance,
    const char* expression,
    const char* file,
    int line) {
    if (std::fabs(actual - expected) > tolerance) {
        std::ostringstream message;
        message << expression << ": actual=" << actual
                << ", expected=" << expected
                << ", tolerance=" << tolerance;
        fail(message.str(), file, line);
    }
}

inline int run_all() {
    int failures = 0;
    for (const TestCase& test : registry()) {
        try {
            test.function();
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        } catch (...) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": unknown exception\n";
        }
    }

    std::cout << registry().size() - static_cast<std::size_t>(failures)
              << '/' << registry().size() << " tests passed\n";
    return failures == 0 ? 0 : 1;
}

}  // namespace vector_search_test

#define VECTOR_SEARCH_TEST_CONCAT_IMPL(left, right) left##right
#define VECTOR_SEARCH_TEST_CONCAT(left, right) \
    VECTOR_SEARCH_TEST_CONCAT_IMPL(left, right)

#define TEST_CASE(name) \
    static void VECTOR_SEARCH_TEST_CONCAT(test_function_, __LINE__)(); \
    static ::vector_search_test::Registrar \
        VECTOR_SEARCH_TEST_CONCAT(test_registrar_, __LINE__)( \
            name, VECTOR_SEARCH_TEST_CONCAT(test_function_, __LINE__)); \
    static void VECTOR_SEARCH_TEST_CONCAT(test_function_, __LINE__)()

#define REQUIRE(condition) \
    ::vector_search_test::require((condition), #condition, __FILE__, __LINE__)

#define REQUIRE_EQ(left, right) \
    ::vector_search_test::require_equal( \
        (left), (right), #left " == " #right, __FILE__, __LINE__)

#define REQUIRE_NEAR(actual, expected, tolerance) \
    ::vector_search_test::require_near( \
        (actual), (expected), (tolerance), \
        #actual " ~= " #expected, __FILE__, __LINE__)
