#ifndef STAN_MATH_REV_MAT_FUN_BUILD_ODE_VARIS_HPP
#define STAN_MATH_REV_MAT_FUN_BUILD_ODE_VARIS_HPP

#include <stan/math/rev/mat/fun/ode_vari.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <algorithm>
#include <ostream>
#include <vector>

namespace stan {
  namespace math {
    /*
     * build_ode_varis takes the CVODES output and wraps it in a 2D Stan array
     * of vars. To avoiding chain for every individual output, we create
     * one giant ode_vari which does the actual adjoint calculation and then
     * num_timesteps * num_states - 1 varis which are just placeholders.
     *
     * Because all the varis are pushed onto the stack at once, whenever any of
     * them are called, all the adjoints at these outputs are ready to be
     * chained back up.
     *
     * By putting all but one vari on the autodiff stack, only one vari gets
     * called and one one backwards ODE solve needs to happen
     *
     * @tparam T_ode_data Convenience template to hide true type of cvodes_data
     * @tparam T_initial Type of initial conditions, can be double or var
     * @tparam T_param Type of parameters, can be double or var
     * @param t0 Initial time for forward ODE solve
     * @param relative_tolerance Relative tolerance of adjoint ODE solve
     * @param absolute_tolerance Absolute tolerance of adjoint ODE solve
     * @param y Output of forward ODE solve
     * @param y0_ Initial conditions for forward ODE solve
     * @param theta_ Parameters for forward ODE solve
     * @param ts Time steps to produce output in forward ODE solve
     * @param cvodes_mem Pointer to CVODES internal memory space
     * @param cvodes_data Pointer to cvodes_ode_data struct that computes
     *   jacobians and such
     */
    template <typename T_ode_data, typename T_initial, typename T_param>
    inline
    std::vector<std::vector<typename stan::return_type<T_initial,
                                                       T_param>::type> >
    build_ode_varis(double t0,
                    double relative_tolerance,
                    double absolute_tolerance,
                    const std::vector<std::vector<double> >& y,
                    const std::vector<T_initial>& y0_,
                    const std::vector<T_param>& theta_,
                    const std::vector<double> &ts,
                    void* cvodes_mem,
                    T_ode_data *cvodes_data) {
      using std::vector;

      const size_t N = y0_.size();

      vector<var> y0(y0_.begin(), y0_.end());
      vector<var> theta(theta_.begin(), theta_.end());
      vector<vector<var> > y_return(y.size(), vector<var>(N, 0));

      vari** non_chaining_varis =
        ChainableStack::memalloc_.alloc_array<vari*>(y.size() * N - 1);

      for (size_t i = 0; i < y.size(); i++) {
        for (size_t j = 0; j < N; j++) {
          // The special vari that will handle the adjoint solve corresponds to
          // the output y[y.size() - 1][N - 1]
          if (i == y.size() - 1 && j == N - 1)
            continue;

          // non_chaining_varis[i * N + j] corresponds to the vari attached to
          // the ode output at time t[i] and state j
          non_chaining_varis[i * N + j] = new vari(y[i][j], false);
        }
      }

      ode_vari<T_ode_data>* ode_vari_ =
        new ode_vari<T_ode_data>(t0,
                                 relative_tolerance,
                                 absolute_tolerance,
                                 y[y.size() - 1][N - 1],
                                 y0,
                                 theta,
                                 ts,
                                 non_chaining_varis,
                                 cvodes_mem,
                                 cvodes_data);

      for (size_t i = 0; i < y.size(); i++)
        for (size_t j = 0; j < N; j++)
          // Inject our special vari at y[y.size() - 1][N - 1]
          if (i == y.size() - 1 && j == N - 1)
            y_return[i][j] = var(ode_vari_);
          else
            y_return[i][j] = var(non_chaining_varis[i * N + j]);

      return y_return;
    }

    /*
     * If theta and y are both doubles, just pass the values through (there's
     * no autodiff to handle here).
     */
    template <typename T_ode_data>
    inline
    std::vector<std::vector<double> >
    build_ode_varis(double t0,
                    double relative_tolerance,
                    double absolute_tolerance,
                    const std::vector<std::vector<double> >& y,
                    const std::vector<double>& y0,
                    const std::vector<double>& theta,
                    const std::vector<double> &ts,
                    void* cvodes_mem,
                    T_ode_data *cvodes_data) {
      return y;
    }
  }
}

#endif
