#include <stan/math/rev/scal.hpp>
#include <gtest/gtest.h>

TEST(AgradRev, value_of) {
  using stan::math::value_of;
  using stan::math::var;

  var a = 5.0;
  EXPECT_FLOAT_EQ(5.0, value_of(a));
  // make sure all work together
  EXPECT_FLOAT_EQ(5.0, value_of(5.0));
  EXPECT_FLOAT_EQ(5.0, value_of(5));
}

TEST(AgradRev, value_of_complex) {
  using stan::math::value_of;
  using stan::math::var;

  std::complex<var> x(1.0, 2.0);
  EXPECT_FLOAT_EQ(1.0, value_of(x).real());
  EXPECT_FLOAT_EQ(2.0, value_of(x).imag());
}
