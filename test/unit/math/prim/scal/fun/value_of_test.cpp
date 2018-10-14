#include <stan/math/prim/scal.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>
#include <limits>

TEST(MathFunctions, value_of) {
  using stan::math::value_of;
  double x = 5.0;
  EXPECT_FLOAT_EQ(5.0, value_of(x));
  EXPECT_FLOAT_EQ(5.0, value_of(5));
}

TEST(MathFunctions, value_of_complex) {
  using stan::math::value_of;
  std::complex<double> x(1.0, 2.0);
  EXPECT_FLOAT_EQ(1.0, value_of(x).real());
  EXPECT_FLOAT_EQ(2.0, value_of(x).imag());
}

TEST(MathFunctions, value_of_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  EXPECT_PRED1(boost::math::isnan<double>, stan::math::value_of(nan));
}
