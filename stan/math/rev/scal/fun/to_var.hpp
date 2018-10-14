#ifndef STAN_MATH_REV_SCAL_FUN_TO_VAR_HPP
#define STAN_MATH_REV_SCAL_FUN_TO_VAR_HPP

#include <stan/math/rev/core.hpp>

namespace stan {
namespace math {

/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a var variable with the input value.
 *
 * @param[in] x A scalar value
 * @return An automatic differentiation variable with the input value.
 */
inline var to_var(double x) { return var(x); }

/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a var variable with the input value.
 *
 * @param[in] x An automatic differentiation variable.
 * @return An automatic differentiation variable with the input value.
 */
inline var to_var(const var& x) { return x; }

/**
 * Converts argument to an automatic differentiation variable.
 *
 * Returns a std::complex<var> variable with the values given in x.
 *
 * @param[in] x A complex value
 * @return An automatic differentiation variable with the input value.
 */
inline std::complex<var> to_var(std::complex<double> x) {
  return std::complex<var>(x);
}

/**
 * Return input if it is already a std::complex<var>.
 *
 * @param[in] x An automatic differentiation variable.
 * @return An automatic differentiation variable with the input value.
 */
inline std::complex<var> to_var(const std::complex<var>& x) { return x; }

}  // namespace math
}  // namespace stan
#endif
