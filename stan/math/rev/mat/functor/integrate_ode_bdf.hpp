#ifndef STAN_MATH_REV_MAT_FUNCTOR_INTEGRATE_ODE_BDF_HPP
#define STAN_MATH_REV_MAT_FUNCTOR_INTEGRATE_ODE_BDF_HPP

#include <stan/math/prim/arr/fun/value_of.hpp>
#include <stan/math/prim/scal/err/check_less.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/arr/err/check_nonzero_size.hpp>
#include <stan/math/prim/arr/err/check_ordered.hpp>
#include <stan/math/rev/scal/meta/is_var.hpp>
#include <stan/math/rev/mat/fun/build_ode_varis.hpp>
#include <stan/math/rev/mat/fun/ode_vari.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/rev/mat/functor/cvodes_utils.hpp>
#include <stan/math/rev/mat/functor/cvodes_ode_data.hpp>
#include <stan/math/rev/arr/fun/decouple_ode_states.hpp>
#include <cvodes/cvodes.h>
#include <cvodes/cvodes_lapack.h>
#include <cvodes/cvodes_dense.h>
#include <nvector/nvector_serial.h>
#include <algorithm>
#include <ostream>
#include <vector>

namespace stan {
  namespace math {
    /**
     * Return the solutions for the specified system of ordinary
     * differential equations given the specified initial state,
     * initial times, times of desired solution, and parameters and
     * data, writing error and warning messages to the specified
     * stream.
     *
     * This function is templated to allow the initial times to be
     * either data or autodiff variables and the parameters to be data
     * or autodiff variables.  The autodiff-based implementation for
     * reverse-mode are defined in namespace <code>stan::math</code>
     * and may be invoked via argument-dependent lookup by including
     * their headers.
     *
     * The solver used is based on the backward differentiation
     * formula which is an implicit numerical integration scheme
     * appropiate for stiff ODE systems.
     *
     * @tparam F type of ODE system function.
     * @tparam T_initial type of scalars for initial values.
     * @tparam T_param type of scalars for parameters.
     * @param[in] f functor for the base ordinary differential equation.
     * @param[in] y0 initial state.
     * @param[in] t0 initial time.
     * @param[in] ts times of the desired solutions, in strictly
     * increasing order, all greater than the initial time.
     * @param[in] theta parameter vector for the ODE.
     * @param[in] x continuous data vector for the ODE.
     * @param[in] x_int integer data vector for the ODE.
     * @param[in, out] msgs the print stream for warning messages.
     * @param[in] relative_tolerance relative tolerance passed to CVODE.
     * @param[in] absolute_tolerance absolute tolerance passed to CVODE.
     * @param[in] max_num_steps maximal number of admissable steps
     * between time-points
     * @return a vector of states, each state being a vector of the
     * same size as the state variable, corresponding to a time in ts.
     */
    template <typename F, typename T_initial, typename T_param>
    std::vector<std::vector<typename stan::return_type<T_initial,
                                                       T_param>::type> >
    integrate_ode_bdf(const F& f,
                      const std::vector<T_initial>& y0,
                      double t0,
                      const std::vector<double>& ts,
                      const std::vector<T_param>& theta,
                      const std::vector<double>& x,
                      const std::vector<int>& x_int,
                      std::ostream* msgs = 0,
                      double relative_tolerance = 1e-10,
                      double absolute_tolerance = 1e-10,
                      long int max_num_steps = 1e8) {  // NOLINT(runtime/int)
      typedef stan::is_var<T_initial> initial_var;
      typedef stan::is_var<T_param> param_var;

      check_finite("integrate_ode_bdf", "initial state", y0);
      check_finite("integrate_ode_bdf", "initial time", t0);
      check_finite("integrate_ode_bdf", "times", ts);
      check_finite("integrate_ode_bdf", "parameter vector", theta);
      check_finite("integrate_ode_bdf", "continuous data", x);
      check_nonzero_size("integrate_ode_bdf", "times", ts);
      check_nonzero_size("integrate_ode_bdf", "initial state", y0);
      check_ordered("integrate_ode_bdf", "times", ts);
      check_less("integrate_ode_bdf", "initial time", t0, ts[0]);
      if (relative_tolerance <= 0)
        invalid_argument("integrate_ode_bdf",
                         "relative_tolerance,", relative_tolerance,
                         "", ", must be greater than 0");
      if (absolute_tolerance <= 0)
        invalid_argument("integrate_ode_bdf",
                         "absolute_tolerance,", absolute_tolerance,
                         "", ", must be greater than 0");
      if (max_num_steps <= 0)
        invalid_argument("integrate_ode_bdf",
                         "max_num_steps,", max_num_steps,
                         "", ", must be greater than 0");

      const size_t N = y0.size();
      const size_t M = theta.size();
      // total number of sensitivities for initial values and params
      const size_t S = (initial_var::value ? N : 0)
        + (param_var::value ? M : 0);
      std::vector<double> state(value_of(y0));
      N_Vector cvodes_state(N_VMake_Serial(N, &state[0]));

      typedef cvodes_ode_data<F, T_initial, T_param> ode_data;
      ode_data *cvodes_data = new ode_data(f, y0, theta, x, x_int, msgs);

      void* cvodes_mem = CVodeCreate(CV_BDF, CV_NEWTON);
      if (cvodes_mem == NULL)
        throw std::runtime_error("CVodeCreate failed to allocate memory");

      std::vector<std::vector<double>> y(ts.size(), std::vector<double>(N, 0));

      try {
        cvodes_check_flag(CVodeInit(cvodes_mem, &ode_data::ode_rhs,
                                    t0, cvodes_state),
                          "CVodeInit");

        // Assign pointer to this as user data
        cvodes_check_flag(CVodeSetUserData(cvodes_mem,
                            reinterpret_cast<void*>(cvodes_data)),
                          "CVodeSetUserData");

        cvodes_set_options(cvodes_mem,
                           relative_tolerance, absolute_tolerance,
                           max_num_steps);

        // for the stiff solvers we need to reserve additional
        // memory and provide a Jacobian function call
        cvodes_check_flag(CVDense(cvodes_mem, N), "CVDense");
        cvodes_check_flag(CVDlsSetDenseJacFn(cvodes_mem,
                                             &ode_data::dense_jacobian),
                          "CVDlsSetDenseJacFn");

        // initialize adjoint sensitivity system of CVODES as needed
        // The 25 here is a new magic parameter that you gotta be careful with
        // This is the number of steps between when the forward mode ODE saves
        // its output. If the number is too high the reverse mode will error
        if (S > 0) {
          cvodes_check_flag(CVodeAdjInit(cvodes_mem, 25, CV_HERMITE),
                            "CVodeAdjInit");
        }

        int ncheck = 0;
        double t_init = t0;
        for (size_t n = 0; n < ts.size(); ++n) {
          double t_final = ts[n];
          if (t_final != t_init) {
            // If we're doing sensitivity analysis, we use the special CVode
            // call CVodeF
            if (S > 0) {
              cvodes_check_flag(CVodeF(cvodes_mem, t_final, cvodes_state,
                                       &t_init, CV_NORMAL, &ncheck),
                                "CVodeF");
            } else {
              cvodes_check_flag(CVode(cvodes_mem, t_final, cvodes_state,
                                      &t_init, CV_NORMAL),
                                "CVode");
            }
          }
          std::copy(state.begin(), state.end(), y[n].begin());
          t_init = t_final;
        }
      } catch (const std::exception& e) {
        N_VDestroy_Serial(cvodes_state);
        CVodeFree(&cvodes_mem);
        delete cvodes_data;
        throw;
      }

      N_VDestroy_Serial(cvodes_state);
      return build_ode_varis(t0, relative_tolerance, absolute_tolerance,
                             y, y0, theta, ts, cvodes_mem, cvodes_data);
    }
  }
}
#endif
