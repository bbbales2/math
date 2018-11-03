#include <stan/math/rev/mat.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <test/unit/math/rev/mat/util.hpp>
#include <test/unit/math/rev/mat/util/test_autodiff.hpp>
#include <vector>

TEST(AgradRevMatrix, complex_constructor) {
  auto func = [](auto b) {
    std::complex<stan::math::var> a(b);
    return a;
  };

  std::complex<double> c1(3.0, 4.0);
  double d = 3.0;

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, d);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1);
}

TEST(AgradRevMatrix, complex_assignment) {
  auto func = [](auto b) {
    std::complex<stan::math::var> a(1.0, 2.0);
    a = b;
    return a;
  };

  std::complex<double> c1(3.0, 4.0);
  double d = 3.0;

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, d);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1);
}

TEST(AgradRevMatrix, complex_member_real) {
  auto func = [](auto a) { return a.real(); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_member_imag) {
  auto func = [](auto a) { return a.imag(); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_member_addition) {
  auto func = [](auto b) {
    std::complex<stan::math::var> a(1.0, 2.0);
    a += b;
    return a;
  };

  std::complex<double> c1(3.0, 4.0);
  double d = 3.0;

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, d);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1);
}

TEST(AgradRevMatrix, complex_member_subtraction) {
  auto func = [](auto b) {
    std::complex<stan::math::var> a(1.0, 2.0);
    a -= b;
    return a;
  };

  std::complex<double> c1(3.0, 4.0);
  double d = 3.0;

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, d);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1);
}

TEST(AgradRevMatrix, complex_member_multiplication) {
  auto func = [](auto b) {
    std::complex<stan::math::var> a(1.0, 2.0);
    a *= b;
    return a;
  };

  std::complex<double> c1(3.0, 4.0);
  double d = 3.0;

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, d);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1);
}

TEST(AgradRevMatrix, complex_member_division) {
  auto func = [](auto b) {
    std::complex<stan::math::var> a(1.0, 2.0);
    a /= b;
    return a;
  };

  std::complex<double> c1(3.0, 4.0);
  double d = 3.0;

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, d);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1);
}

TEST(AgradRevMatrix, complex_addition) {
  auto func = [](auto a, auto b) { return a + b; };

  std::complex<double> c1(1.0, 2.0), c2(3.0, 4.0);
  double d = 3.0;

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1, d);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, d, c1);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1, c2);
}

TEST(AgradRevMatrix, complex_subtraction) {
  auto func = [](auto a, auto b) { return a - b; };

  std::complex<double> c1(1.0, 2.0), c2(3.0, 4.0);
  double d = 3.0;

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1, d);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, d, c1);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1, c2);
}

TEST(AgradRevMatrix, complex_multiplication) {
  auto func = [](auto a, auto b) { return a * b; };

  std::complex<double> c1(1.0, 2.0), c2(3.0, 4.0);
  double d = 3.0;

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1, d);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, d, c1);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1, c2);
}

TEST(AgradRevMatrix, complex_division) {
  auto func = [](auto a, auto b) { return a / b; };

  std::complex<double> c1(1.0, 2.0), c2(3.0, 4.0);
  double d = 3.0;

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1, d);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, d, c1);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1, c2);
}

TEST(AgradRevMatrix, complex_real) {
  auto func = [](auto a) { return std::real(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_imag) {
  auto func = [](auto a) { return std::imag(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_abs) {
  auto func = [](auto a) { return std::abs(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_arg) {
  auto func = [](auto a) { return std::arg(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_norm) {
  auto func = [](auto a) { return std::norm(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_conj) {
  auto func = [](auto a) { return std::conj(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_proj) {
  auto func = [](auto a) { return std::proj(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

// Failing
TEST(AgradRevMatrix, complex_polar) {
  auto func = [](auto a, auto b) { return std::polar(a, b); };

  double a = 1.0;
  double b = 2.0;

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, a, b);
}

TEST(AgradRevMatrix, complex_exp) {
  auto func = [](auto a) { return std::exp(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_log) {
  auto func = [](auto a) { return std::log(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_log10) {
  auto func = [](auto a) { return std::log10(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

// Failing
TEST(AgradRevMatrix, complex_pow) {
  auto func = [](auto a, auto b) { return std::pow(a, b); };

  std::complex<double> c1(1.0, 2.0), c2(-1.0, -2.0);
  double d = 4.0;

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, d, c2);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1, d);
  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c1, c2);
}

TEST(AgradRevMatrix, complex_sqrt) {
  auto func = [](auto a) { return std::sqrt(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_sin) {
  auto func = [](auto a) { return std::sin(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_cos) {
  auto func = [](auto a) { return std::cos(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_tan) {
  auto func = [](auto a) { return std::tan(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_asin) {
  auto func = [](auto a) { return std::asin(a); };

  std::complex<double> c = std::sin(std::complex<double>(1.0, 2.0));

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_acos) {
  auto func = [](auto a) { return std::acos(a); };

  std::complex<double> c = std::cos(std::complex<double>(1.0, 2.0));

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_atan) {
  auto func = [](auto a) { return std::atan(a); };

  std::complex<double> c = std::tan(std::complex<double>(1.0, 2.0));

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_sinh) {
  auto func = [](auto a) { return std::sinh(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_cosh) {
  auto func = [](auto a) { return std::cosh(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_tanh) {
  auto func = [](auto a) { return std::tanh(a); };

  std::complex<double> c(1.0, 2.0);

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_asinh) {
  auto func = [](auto a) { return std::asinh(a); };

  std::complex<double> c = std::sinh(std::complex<double>(1.0, 2.0));

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_acosh) {
  auto func = [](auto a) { return std::acosh(a); };

  std::complex<double> c = std::cosh(std::complex<double>(1.0, 2.0));

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}

TEST(AgradRevMatrix, complex_atanh) {
  auto func = [](auto a) { return std::atanh(a); };

  std::complex<double> c = std::tanh(std::complex<double>(1.0, 2.0));

  stan::math::test::test_autodiff(func, 1e-6, 1e-6, c);
}
