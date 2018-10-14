#include <stan/math/rev/scal.hpp>
#include <gtest/gtest.h>

TEST(AgradRev, to_var) {
  using stan::math::value_of;
  using stan::math::var;

  double a = 5.0;
  auto b = stan::math::to_var(a);
  auto c = stan::math::to_var(b);
  EXPECT_TRUE((std::is_same<stan::math::var, decltype(b)>::value));
  EXPECT_TRUE((std::is_same<stan::math::var, decltype(c)>::value));
  EXPECT_FLOAT_EQ(a, value_of(b));
  EXPECT_FLOAT_EQ(a, value_of(c));
}

TEST(AgradRev, value_of_complex) {
  using stan::math::value_of;
  using stan::math::var;

  std::complex<double> a(1.0, 2.0);
  auto b = stan::math::to_var(a);
  auto c = stan::math::to_var(b);
  EXPECT_TRUE((std::is_same<std::complex<stan::math::var>, decltype(b)>::value));
  EXPECT_TRUE((std::is_same<std::complex<stan::math::var>, decltype(c)>::value));
  EXPECT_FLOAT_EQ(a.real(), value_of(b.real()));
  EXPECT_FLOAT_EQ(a.imag(), value_of(b.imag()));
  EXPECT_FLOAT_EQ(a.real(), value_of(c.real()));
  EXPECT_FLOAT_EQ(a.imag(), value_of(c.imag()));
}
