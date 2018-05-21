#include <stan/math/mix/mat.hpp>
#include <gtest/gtest.h>
#include <test/unit/math/rev/mat/fun/util.hpp>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

using Eigen::Dynamic;
using Eigen::Matrix;

struct fun0 {
  template <typename T>
  inline T operator()(const T& x) const {
    return 5.0 * x * x * x;
  }
};

// fun1(x, y) = (x^2 * y) + (3 * y^2)
struct fun1 {
  template <typename T>
  inline T operator()(const Matrix<T, Dynamic, 1>& x) const {
    return x(0) * x(0) * x(1) + 3.0 * x(1) * x(1);
  }
};

// fun2: R^2 --> R^2 | (x, y) --> [(x + x), (3 * x * y)]
struct fun2 {
  template <typename T>
  inline Matrix<T, Dynamic, 1> operator()(
      const Matrix<T, Dynamic, 1>& x) const {
    Matrix<T, Dynamic, 1> z(2);
    z << x(0) + x(0), 3 * x(0) * x(1);
    return z;
  }
};

struct norm_functor {
  template <typename T>
  inline T operator()(
      const Eigen::Matrix<T, Eigen::Dynamic, 1>& inp_vec) const {
    return stan::math::normal_log(inp_vec(0), inp_vec(1), inp_vec(2));
  }
};

TEST(AgradAutoDiff, derivative) {
  fun0 f;
  double x = 7;
  double fx;
  double d;
  stan::math::derivative(f, x, fx, d);
  EXPECT_FLOAT_EQ(fx, 5 * 7 * 7 * 7);
  EXPECT_FLOAT_EQ(d, 5 * 3 * 7 * 7);
}

TEST(AgradAutoDiff, partialDerivative) {
  fun1 f;
  Matrix<double, Dynamic, 1> x(2);
  x << 5, 7;

  double fx;
  double d;
  stan::math::partial_derivative(f, x, 0, fx, d);
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7, fx);
  EXPECT_FLOAT_EQ(2 * 5 * 7, d);

  double fx2;
  double d2;
  stan::math::partial_derivative(f, x, 1, fx2, d2);
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7, fx);
  EXPECT_FLOAT_EQ(5 * 5 + 3 * 2 * 7, d2);
}

TEST(AgradAutoDiff, gradientDotVector) {
  using stan::math::var;
  fun1 f;
  Matrix<double, Dynamic, 1> x(2);
  x << 5, 7;
  Matrix<double, Dynamic, 1> v(2);
  v << 11, 13;
  double fx;
  double grad_fx_dot_v;
  stan::math::gradient_dot_vector(f, x, v, fx, grad_fx_dot_v);

  double fx_expected;
  Matrix<double, Dynamic, 1> grad_fx;
  stan::math::gradient(f, x, fx_expected, grad_fx);
  double grad_fx_dot_v_expected = grad_fx.dot(v);

  EXPECT_FLOAT_EQ(grad_fx_dot_v_expected, grad_fx_dot_v);
}
TEST(AgradAutoDiff, hessianTimesVector) {
  using stan::math::hessian_times_vector;

  fun1 f;

  Matrix<double, Dynamic, 1> x(2);
  x << 2, -3;

  Matrix<double, Dynamic, 1> v(2);
  v << 8, 5;

  Matrix<double, Dynamic, 1> Hv;
  double fx;
  stan::math::hessian_times_vector(f, x, v, fx, Hv);

  EXPECT_FLOAT_EQ(2 * 2 * -3 + 3.0 * -3 * -3, fx);

  EXPECT_EQ(2, Hv.size());
  EXPECT_FLOAT_EQ(2 * x(1) * v(0) + 2 * x(0) * v(1), Hv(0));
  EXPECT_FLOAT_EQ(2 * x(0) * v(0) + 6 * v(1), Hv(1));
}

TEST(AgradAutoDiff, jacobian) {
  using stan::math::jacobian;

  fun2 f;
  Matrix<double, Dynamic, 1> x(2);
  x << 2, -3;

  Matrix<double, Dynamic, 1> fx;
  Matrix<double, Dynamic, Dynamic> J;
  jacobian(f, x, fx, J);

  EXPECT_EQ(2, fx.size());
  EXPECT_FLOAT_EQ(2 * 2, fx(0));
  EXPECT_FLOAT_EQ(3 * 2 * -3, fx(1));

  EXPECT_FLOAT_EQ(2, J(0, 0));
  EXPECT_FLOAT_EQ(-9, J(1, 0));
  EXPECT_FLOAT_EQ(0, J(0, 1));
  EXPECT_FLOAT_EQ(6, J(1, 1));

  Matrix<double, Dynamic, 1> fx_rev;
  Matrix<double, Dynamic, Dynamic> J_rev;
  jacobian<double>(f, x, fx_rev, J_rev);

  EXPECT_EQ(2, fx_rev.size());
  EXPECT_FLOAT_EQ(2 * 2, fx_rev(0));
  EXPECT_FLOAT_EQ(3 * 2 * -3, fx_rev(1));

  EXPECT_FLOAT_EQ(2, J_rev(0, 0));
  EXPECT_FLOAT_EQ(-9, J_rev(1, 0));
  EXPECT_FLOAT_EQ(0, J_rev(0, 1));
  EXPECT_FLOAT_EQ(6, J_rev(1, 1));
}

TEST(AgradAutoDiff, hessian) {
  fun1 f;
  Matrix<double, Dynamic, 1> x(2);
  x << 5, 7;
  double fx(0);
  Matrix<double, Dynamic, 1> grad;
  Matrix<double, Dynamic, Dynamic> H;
  stan::math::hessian(f, x, fx, grad, H);

  // x^2 * y + 3 * y^2
  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7, fx);

  EXPECT_FLOAT_EQ(2, grad.size());
  EXPECT_FLOAT_EQ(2 * x(0) * x(1), grad(0));
  EXPECT_FLOAT_EQ(x(0) * x(0) + 3 * 2 * x(1), grad(1));

  EXPECT_EQ(2, H.rows());
  EXPECT_EQ(2, H.cols());
  EXPECT_FLOAT_EQ(2 * 7, H(0, 0));
  EXPECT_FLOAT_EQ(2 * 5, H(0, 1));
  EXPECT_FLOAT_EQ(2 * 5, H(1, 0));
  EXPECT_FLOAT_EQ(2 * 3, H(1, 1));

  double fx2;
  Matrix<double, Dynamic, 1> grad2;
  Matrix<double, Dynamic, Dynamic> H2;
  stan::math::hessian<double>(f, x, fx2, grad2, H2);

  EXPECT_FLOAT_EQ(5 * 5 * 7 + 3 * 7 * 7, fx2);

  EXPECT_FLOAT_EQ(2, grad2.size());
  EXPECT_FLOAT_EQ(2 * x(0) * x(1), grad2(0));
  EXPECT_FLOAT_EQ(x(0) * x(0) + 3 * 2 * x(1), grad2(1));

  EXPECT_EQ(2, H2.rows());
  EXPECT_EQ(2, H2.cols());
  EXPECT_FLOAT_EQ(2 * 7, H2(0, 0));
  EXPECT_FLOAT_EQ(2 * 5, H2(0, 1));
  EXPECT_FLOAT_EQ(2 * 5, H2(1, 0));
  EXPECT_FLOAT_EQ(2 * 3, H2(1, 1));
}

TEST(AgradAutoDiff, GradientTraceMatrixTimesHessian) {
  Matrix<double, Dynamic, Dynamic> M(2, 2);
  M << 11, 13, 17, 23;
  fun1 f;
  Matrix<double, Dynamic, 1> x(2);
  x << 5, 7;
  Matrix<double, Dynamic, 1> grad_tr_MH;
  stan::math::grad_tr_mat_times_hessian(f, x, M, grad_tr_MH);

  EXPECT_EQ(2, grad_tr_MH.size());
  EXPECT_FLOAT_EQ(60, grad_tr_MH(0));
  EXPECT_FLOAT_EQ(22, grad_tr_MH(1));
}

TEST(AgradAutoDiff, GradientHessian) {
  norm_functor log_normal_density;
  third_order_mixed mixed_third_poly;

  Matrix<double, Dynamic, 1> normal_eval_vec(3);
  Matrix<double, Dynamic, 1> poly_eval_vec(3);

  normal_eval_vec << 0.7, 0.5, 0.9;
  poly_eval_vec << 1.5, 7.1, 3.1;

  double normal_eval_agrad;
  double poly_eval_agrad;

  double normal_eval_agrad_hessian;
  double poly_eval_agrad_hessian;

  double normal_eval_analytic;
  double poly_eval_analytic;

  Matrix<double, Dynamic, Dynamic> norm_hess_agrad;
  Matrix<double, Dynamic, Dynamic> poly_hess_agrad;

  Matrix<double, Dynamic, Dynamic> norm_hess_agrad_hessian;
  Matrix<double, Dynamic, Dynamic> poly_hess_agrad_hessian;

  Matrix<double, Dynamic, 1> norm_grad_agrad_hessian;
  Matrix<double, Dynamic, 1> poly_grad_agrad_hessian;

  stan::math::hessian(log_normal_density, normal_eval_vec,
                      normal_eval_agrad_hessian, norm_grad_agrad_hessian,
                      norm_hess_agrad_hessian);

  stan::math::hessian(mixed_third_poly, poly_eval_vec, poly_eval_agrad_hessian,
                      poly_grad_agrad_hessian, poly_hess_agrad_hessian);

  Matrix<double, Dynamic, Dynamic> norm_hess_analytic;
  Matrix<double, Dynamic, Dynamic> poly_hess_analytic;

  std::vector<Matrix<double, Dynamic, Dynamic>> norm_grad_hess_agrad;
  std::vector<Matrix<double, Dynamic, Dynamic>> poly_grad_hess_agrad;

  std::vector<Matrix<double, Dynamic, Dynamic>> norm_grad_hess_analytic;
  std::vector<Matrix<double, Dynamic, Dynamic>> poly_grad_hess_analytic;

  normal_eval_analytic = log_normal_density(normal_eval_vec);
  poly_eval_analytic = mixed_third_poly(poly_eval_vec);

  stan::math::grad_hessian(log_normal_density, normal_eval_vec,
                           normal_eval_agrad, norm_hess_agrad,
                           norm_grad_hess_agrad);
  stan::math::grad_hessian(mixed_third_poly, poly_eval_vec, poly_eval_agrad,
                           poly_hess_agrad, poly_grad_hess_agrad);
  norm_hess_analytic = norm_hess(normal_eval_vec);
  poly_hess_analytic = third_order_mixed_hess(poly_eval_vec);

  norm_grad_hess_analytic = norm_grad_hess(normal_eval_vec);
  poly_grad_hess_analytic = third_order_mixed_grad_hess(poly_eval_vec);

  EXPECT_FLOAT_EQ(normal_eval_analytic, normal_eval_agrad);
  EXPECT_FLOAT_EQ(poly_eval_analytic, poly_eval_agrad);

  for (size_t i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) {
        if (i == 0) {
          EXPECT_FLOAT_EQ(norm_hess_agrad_hessian(j, k),
                          norm_hess_analytic(j, k));
          EXPECT_FLOAT_EQ(poly_hess_agrad_hessian(j, k),
                          poly_hess_analytic(j, k));
          EXPECT_FLOAT_EQ(norm_hess_analytic(j, k), norm_hess_agrad(j, k));
          EXPECT_FLOAT_EQ(poly_hess_analytic(j, k), poly_hess_agrad(j, k));
        }
        EXPECT_FLOAT_EQ(norm_grad_hess_analytic[i](j, k),
                        norm_grad_hess_agrad[i](j, k));
        EXPECT_FLOAT_EQ(poly_grad_hess_analytic[i](j, k),
                        poly_grad_hess_agrad[i](j, k));
      }
}

{  // begin scope of using declarations for static_assert checks
  // do not use anonymous namespace (unique per TU & injects names)
  using cScal = std::complex<double>;
  using cT = Eigen::VectorBlock<
      Eigen::Block<Eigen::Matrix<std::complex<double>, 2, 2, 0, 2, 2>, 1, 2,
                   false>,
      -1>;
  static_assert(!stan::math::internal::is_cplx_or_arith_v<cT>);
  // ensure Eigen still thinks complex VectorBlocks aren't scalar operands
  static_assert(
      !Eigen::internal::has_ReturnType<Eigen::ScalarBinaryOpTraits<
          cScal, cT, Eigen::internal::scalar_product_op<cScal, cT>>>::value);
}  // end scope of using declarations for static_assert checks

// helper functions for variable conversion to double
double val_help(stan::math::var const v) { return v.val(); }
template <class T>
T val_help(stan::math::fvar<T> const& f) {
  return f.val();
}

template <class S, class V>
S val(V const& v) {
  if constexpr (std::is_same_v<S, V>)
    return v;
  return val<S>(val_help(v));
}

template <class S, class V>
std::complex<S> val(std::complex<V> const& v) {
  if constexpr (std::is_same_v<S, V>)
    return v;
  return std::complex<S>(val<S>(v.real()), val<S>(v.imag()));
}

TEST(AgradAutoDiff, ComplexEigenvalueOfRotationGradientHessian) {
  auto tol([](auto d) {
    return pow(2., -53. / 2.) * (fabs(d) + 1.0);
  });  // tolerance
  auto equal(
      [tol](auto l, auto r) { return fabs(l - r) < tol(l); });  // equality
  auto dbl([&](auto e) { return val<double>(e); });

  // return an eigenvalue of rotation by angle alpha
  auto rotation_eigenvalue([equal, dbl](auto alpha) {
    auto c(cos(alpha));  // cosine of rotation angle alpha
    auto s(sin(alpha));  // sine of rotation angle alpha
    // rotation matrix
    auto r((Eigen::Matrix<decltype(c), 2, 2>() << c, -s, s, c).finished());
    Eigen::EigenSolver<decltype(r)> es(r);           // complex eigensolution
    auto vtr(es.eigenvectors().transpose().eval());  // eigenvectors v^T
    // rv=vD; (rv)^T=(vD)^T; v^T r^T = D^T v^T = D v^T; r^T = v^T^-1 D v^T
    auto r_check((vtr.fullPivLu().solve(es.eigenvalues().asDiagonal() * vtr))
                     .transpose()
                     .eval());  // ScalarType=std::complex<...>
    auto r_err_mat((r.unaryExpr(dbl) - r_check.unaryExpr(dbl)).eval());
    auto r_err(fabs(r_err_mat.norm()));  // check re-composition error
    EXPECT_TRUE(equal(r_err, 0.0));      // tight tolerance residual
    return es.eigenvalues()[0];          // return whatever eigenvalue is first
  });                                    // end of rotation_eigenvalue

  auto f([&](auto x) {  // objective critical min=0 when real = imag
    auto rv(rotation_eigenvalue(x[0]));    // cplx eigenvalue of rotation
    return pow(rv.real() - rv.imag(), 2);  // delta^2 of real and imag parts
  });                                      // end of f

  auto x((Eigen::VectorXd(1) << 3. * M_PI / 8.).finished());  // note: cplx eig
  double fx(f(x));  // objective function value
  EXPECT_FALSE(equal(fx, 0.));
 auto rv_ini(rotation_eigenvalue(x[0]);
 EXPECT_FALSE(equal(rv_ini.real(), rv_ini.imag()));
 auto g((Eigen::VectorXd(1) << 1.).finished());  // gradient
 auto h((Eigen::MatrixXd(1, 1) << 0.).finished());  // hessian
 auto fx_old(fx+2.*tol(fx));  // old f(x) value offset to enter loop
 for (auto i(0u); i < 9u && !(equal(fx, fx_old) && equal(g.norm(), 0.)); i++) {
    fx_old = fx;
    stan::math::hessian(f, x, fx, g, h);
    x -= h.fullPivLu().solve(g);  // newton's method for minimization
 }
 EXPECT_TRUE(equal(x[0], M_PI/4.));
 EXPECT_TRUE(equal(fx, 0.));
 auto rv_fin(rotation_eigenvalue(x[0]);
 EXPECT_TRUE(equal(rv_fin.real(), rv_fin.imag()));
}
