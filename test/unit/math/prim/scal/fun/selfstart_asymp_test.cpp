#include <stan/math.hpp>
#include <stan/math/prim/scal/fun/selfstart_asymp.hpp>
#include <gtest/gtest.h>
#include <vector>

double test_selfstart_asymp(double x, double Asym, double resp0, 
                            double exp_lrc) {
  return Asym + (resp0 - Asym) * std::exp(-exp_lrc * x);
}

std::vector<double> test_selfstart_asymp(const std::vector<double>& x, 
                                         double Asym, double resp0, 
                                         double lrc) {
  std::vector<double> value(x.size());
  double exp_lrc = std::exp(lrc);  

  for (std::size_t i = 0; i < x.size(); ++i) {
    value[i] = test_selfstart_asymp(x[i], Asym, resp0, exp_lrc);
  }
  return value; 
}

std::vector<double> test_selfstart_offset_asymp(const std::vector<double> x,
                                                double Asym, double lrc,
                                                double c0) {
  std::vector<double> value(x.size());
  double exp_lrc = std::exp(lrc);

  for (std::size_t i = 0; i < x.size(); ++i)
    value[i] = Asym * (1 - std::exp(-exp_lrc * (x[i] - c0)));

  return value;
}

std::vector<double> test_selfstart_origin_asymp(const std::vector<double> x,
                                                double Asym, double lrc) {
  std::vector<double> value(x.size());
  double exp_lrc = std::exp(lrc);

  for (std::size_t i = 0; i < x.size(); ++i)
    value[i] = Asym * (1 - std::exp(-exp_lrc * x[i]));

  return value;
}

TEST(MathFunctions, selfstart_asymp) {
  using stan::math::selfstart_asymp;

  double Asym = 100;
  double resp0 = -8.5;
  double exp_lrc = std::exp(-3.2);
  double lrc = -3.2;

  EXPECT_FLOAT_EQ(test_selfstart_asymp(0, Asym, resp0, exp_lrc),
                   selfstart_asymp(0, Asym, resp0, exp_lrc));
  EXPECT_FLOAT_EQ(test_selfstart_asymp(-1, Asym, resp0, exp_lrc),
                   selfstart_asymp(-1, Asym, resp0, exp_lrc)); 
  EXPECT_FLOAT_EQ(test_selfstart_asymp(3.14, Asym, resp0, exp_lrc),
                   selfstart_asymp(3.14, Asym, resp0, exp_lrc));
  
  std::vector<double> input;
  input.push_back(0);
  input.push_back(-1);
  input.push_back(3.14);

  std::vector<double> test_results = 
    test_selfstart_asymp(input, Asym, resp0, lrc);

  std::vector<double> fun_results =
    selfstart_asymp(input, Asym, resp0, lrc);

  EXPECT_EQ(test_results.size(), fun_results.size());
  for (std::size_t i = 0; i < test_results.size(); ++i) 
    EXPECT_FLOAT_EQ(test_results[i], fun_results[i]);  
 
}

TEST(MathFunctions, selfstart_offset_asymp) {
  using stan::math::selfstart_offset_asymp;

  double Asym = 32;
  double lrc = -4;
  //double exp_lrc = std::exp(lrc);
  double c0 = 43;

  std::vector<double> input;
  input.push_back(0);
  input.push_back(-1);
  input.push_back(3.14);

  std::vector<double> test_results =
    test_selfstart_offset_asymp(input, Asym, lrc, c0);

  std::vector<double> fun_results =
    selfstart_offset_asymp(input, Asym, lrc, c0);

  EXPECT_EQ(test_results.size(), fun_results.size());

  for (std::size_t i = 0; i < test_results.size(); ++i)
    EXPECT_FLOAT_EQ(test_results[i], fun_results[i]);
}

TEST(MathFunctions, selfstart_origin_asymp) {
  using stan::math::selfstart_origin_asymp;

  double Asym = 32;
  double lrc = -4;
  //double exp_lrc = std::exp(lrc);

  std::vector<double> input;
  input.push_back(0);
  input.push_back(-1);
  input.push_back(3.14);

  std::vector<double> test_results =
    test_selfstart_origin_asymp(input, Asym, lrc);

  std::vector<double> fun_results =
    selfstart_origin_asymp(input, Asym, lrc);

  EXPECT_EQ(test_results.size(), fun_results.size());

  for (std::size_t i = 0; i < test_results.size(); ++i)
    EXPECT_FLOAT_EQ(test_results[i], fun_results[i]);
}