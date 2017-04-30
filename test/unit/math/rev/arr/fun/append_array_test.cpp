#include <stan/math/rev/arr.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/arr/util.hpp>


TEST(MathFunctions, append_array_var) {
  std::vector<double> x(3);
  std::vector<stan::math::var> y(2), result;

  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = 3.0;
  y[0] = 0.5;
  y[1] = 4.0;

  EXPECT_NO_THROW(result = stan::math::append_array(x, y));
  EXPECT_EQ(5, result.size());
  EXPECT_FLOAT_EQ(1.0, result[0].val());
  EXPECT_FLOAT_EQ(2.0, result[1].val());
  EXPECT_FLOAT_EQ(3.0, result[2].val());
  EXPECT_FLOAT_EQ(0.5, result[3].val());
  EXPECT_FLOAT_EQ(4.0, result[4].val());

  std::vector<double> dr;

  result[0].grad(y, dr);
  EXPECT_FLOAT_EQ(0.0, dr[0]);
  EXPECT_FLOAT_EQ(0.0, dr[1]);
  result[1].grad(y, dr);
  EXPECT_FLOAT_EQ(0.0, dr[0]);
  EXPECT_FLOAT_EQ(0.0, dr[1]);
  result[2].grad(y, dr);
  EXPECT_FLOAT_EQ(0.0, dr[0]);
  EXPECT_FLOAT_EQ(0.0, dr[1]);
  result[3].grad(y, dr);
  EXPECT_FLOAT_EQ(1.0, dr[0]);
  EXPECT_FLOAT_EQ(0.0, dr[1]);
  result[4].grad(y, dr);
  EXPECT_FLOAT_EQ(0.0, dr[0]);
  EXPECT_FLOAT_EQ(1.0, dr[1]);
}

/*TEST(AgradRev, sum_std_vector) {
  using stan::math::sum;
  using std::vector;
  using stan::math::var;

  vector<var> x;
  for (size_t i = 0; i < 6; ++i)
    x.push_back(i + 1);
  
  var fx = 3.7 * sum(x);
  EXPECT_FLOAT_EQ(3.7 * 21.0, fx.val());

  vector<double> gx;
  fx.grad(x, gx);
  EXPECT_EQ(6, gx.size());
  for (size_t i = 0; i < 6; ++i)
    EXPECT_FLOAT_EQ(3.7, gx[i]);

  x = vector<var>();
  EXPECT_FLOAT_EQ(0.0, sum(x).val());
}

TEST(AgradRev, check_varis_on_stack) {
  std::vector<stan::math::var> x;
  for (size_t i = 0; i < 6; ++i)
    x.push_back(i + 1);
  test::check_varis_on_stack(stan::math::sum(x));
  }*/
