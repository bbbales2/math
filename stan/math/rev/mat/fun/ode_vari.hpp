#ifndef STAN_MATH_REV_MAT_FUN_ODE_VARI_HPP
#define STAN_MATH_REV_MAT_FUN_ODE_VARI_HPP

#include <stan/math/rev/mat/functor/cvodes_ode_data.hpp>
#include <cvodes/cvodes.h>
#include <cvodes/cvodes_lapack.h>
#include <cvodes/cvodes_dense.h>
#include <nvector/nvector_serial.h>
#include <ostream>
#include <vector>

namespace stan {
  namespace math {
    /*
     * ode_vari is the special vari that will handle running the adjoint ODE
     * for all the other varis
     *
     * @tparam T_ode_data Type of cvodes_ode_data structure
     */
    template <typename T_ode_data>
    class ode_vari : public vari {
    protected:
      double t0_;
      int N_;
      int M_;
      double relative_tolerance_;
      double absolute_tolerance_;
      vari** initial_v_;
      vari** theta_v_;
      std::vector<double> ts_;
      vari** non_chaining_varis_;
      void* cvodes_mem_;
      T_ode_data* cvodes_data_;

    public:
    /*
     * ode_vari is the special vari that handles running the adjoint ODE.
     * For each ODE solve, there is only one ode_vari
     *
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
      explicit ode_vari(double t0,
                        double relative_tolerance,
                        double absolute_tolerance,
                        double value,
                        const std::vector<var> &initial,
                        const std::vector<var> &theta,
                        const std::vector<double> &ts,
                        vari** non_chaining_varis,
                        void* cvodes_mem,
                        T_ode_data *cvodes_data)
        : vari(value),
          t0_(t0),
          N_(initial.size()),
          M_(theta.size()),
          relative_tolerance_(relative_tolerance),
          absolute_tolerance_(absolute_tolerance),
          initial_v_(reinterpret_cast<vari**>(ChainableStack::memalloc_
                                              .alloc(initial.size() *
                                                     sizeof(vari *)))),
          theta_v_(reinterpret_cast<vari**>(ChainableStack::memalloc_
                                            .alloc(theta.size() *
                                                   sizeof(vari *)))),
          ts_(ts),
          non_chaining_varis_(non_chaining_varis),
          cvodes_mem_(cvodes_mem),
          cvodes_data_(cvodes_data) {
        // These are the input parameters, so we'll need to increment these
        // adjoints
        for (size_t i = 0; i < N_; i++)
          initial_v_[i] = initial[i].vi_;
        for (size_t i = 0; i < M_; i++)
          theta_v_[i] = theta[i].vi_;
      }

      virtual void chain() {
        // std::cout << "chain" << std::endl; <-- Good way to verify it's only
        //  being called once
        N_Vector cvodes_state_sens(N_VNew_Serial(N_ + M_));
        N_VConst(0.0, cvodes_state_sens);

        try {
          int indexB;
          // This is all boilerplate CVODES setting up the adjoint ODE to solve
          // CV_ADAMS seemed to work better than CV_BDF on the toy problem
          // I was playing with.
          cvodes_check_flag(CVodeCreateB
                            (cvodes_mem_, CV_ADAMS, CV_NEWTON, &indexB),
                            "CVodeCreateB");

          cvodes_check_flag(CVodeSetUserDataB
                            (cvodes_mem_, indexB,
                             reinterpret_cast<void*>(cvodes_data_)),
                            "CVodeSetUserDataB");

          // The ode_rhs_adj_sense functions passed in here cause problems with
          // the autodiff stack (they can cause reallocations of the internal
          // vectors and cause segfaults)
          cvodes_check_flag(CVodeInitB(cvodes_mem_,
                                       indexB,
                                       &T_ode_data::ode_rhs_adj_sens,
                                       ts_.back(),
                                       cvodes_state_sens),
                            "CVodeInitB");

          cvodes_check_flag(CVodeSStolerancesB(cvodes_mem_,
                                               indexB,
                                               relative_tolerance_,
                                               absolute_tolerance_),
                            "CVodeSStolerancesB");

          cvodes_check_flag(CVDenseB(cvodes_mem_, indexB, N_ + M_),
                            "CVDenseB");

          // The same autodiff issue that applies to doe_rhs_adj_sense applies
          // here
          cvodes_check_flag(CVDlsSetDenseJacFnB
                            (cvodes_mem_, indexB,
                             &T_ode_data::dense_jacobian_adj),
                            "CVDlsSetDenseJacFnB");

          // At every time step, collect the adjoints from the output
          // variables and re-initialize the solver
          for (int i = ts_.size() - 1; i >= 0; --i) {
            // Take in the adjoints from all the output variables at this point
            // in time
            for (int j = 0; j < N_; j++) {
              if (i == ts_.size() - 1 && j == N_ - 1) {
                NV_Ith_S(cvodes_state_sens, j) += adj_;
              } else {
                NV_Ith_S(cvodes_state_sens, j) +=
                  non_chaining_varis_[i * N_ + j]->adj_;
              }
            }

            cvodes_check_flag(CVodeReInitB(cvodes_mem_, indexB, ts_[i],
                                           cvodes_state_sens),
                              "CVodeB");

            cvodes_check_flag(CVodeB(cvodes_mem_, (i > 0) ? ts_[i - 1] : t0_,
                                     CV_NORMAL),
                              "CVodeB");

            // Currently unused, but I should probably check it. CVodeGetB sets
            // it to the time that the ode successfully integrated back to.
            // This should be equal to (i > 0) ? ts_[i - 1] : t0_
            double tret;

            cvodes_check_flag(CVodeGetB(cvodes_mem_, indexB, &tret,
                                        cvodes_state_sens),
                              "CVodeGetB");
          }

          // After integrating all the way back to t0, we finally have the
          // the adjoints we wanted
          // These are the dlog_density / d(initial_conditions[s]) adjoints
          for (size_t s = 0; s < N_; s++) {
            initial_v_[s]->adj_ += NV_Ith_S(cvodes_state_sens, s);
          }

          // These are the dlog_density / d(parameters[s]) adjoints
          for (size_t s = 0; s < M_; s++) {
            theta_v_[s]->adj_ += NV_Ith_S(cvodes_state_sens, N_ + s);
          }
        } catch (const std::exception& e) {
          N_VDestroy_Serial(cvodes_state_sens);
          throw;
        }

        // At some point in the future I should free memory
        //  N_VDestroy_Serial(cvodes_state_sens);
        //  CVodeFree(&cvodes_mem_);
        //  delete cvodes_data_;
      }
    };
  }
}
#endif
