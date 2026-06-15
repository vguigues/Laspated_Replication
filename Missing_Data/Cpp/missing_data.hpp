#include <xtensor-blas/xlinalg.hpp>

#include "laspated.h"

class MissingLambdaRegularizedModel {
 public:
  std::string name = "Regularized";
  ulong C;
  ulong D;
  ulong T;
  ulong R;

  xt::xarray<int> nb_observations;
  xt::xarray<int> nb_arrivals;
  xt::xarray<int> nb_missing_arrivals;
  std::vector<std::vector<std::pair<int, int>>> groups;
  xt::xarray<int> which_group;
  xt::xarray<double> durations;
  xt::xarray<double> alpha;
  std::vector<std::vector<int>> neighbors;
  std::vector<double> weight;
  laspated::Param& param;

  MissingLambdaRegularizedModel(
      xt::xarray<int>& a_nb_observations, xt::xarray<int>& a_nb_arrivals,
      xt::xarray<int>& a_nb_missing_arrivals, xt::xarray<double>& alphas,
      std::vector<double>& weights,
      std::vector<std::vector<std::pair<int, int>>>& a_groups,
      std::vector<std::vector<int>>& a_neighbors,
      xt::xarray<double>& a_durations, laspated::Param& a_param)
      : param(a_param) {
    nb_observations = a_nb_observations;
    C = nb_observations.shape(0);
    D = nb_observations.shape(1);
    T = nb_observations.shape(2);
    R = nb_observations.shape(3);
    nb_arrivals = a_nb_arrivals;
    nb_missing_arrivals = a_nb_missing_arrivals;
    groups = a_groups;
    xt::xarray<int> which_group = -1 * xt::ones<int>({D, T});
    for (int g = 0; g < groups.size(); ++g) {
      for (auto elem : groups[g]) {
        int d = elem.first;
        int t = elem.second;
        which_group(d, t) = g;
      }
    }
    durations = a_durations;
    alpha = alphas;
    neighbors = a_neighbors;
    weight = weights;
  }

  double f(xt::xarray<double>& x) {
    double obj = 0.0;
    double first_s_weight = alpha(0, 0);
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int r = 0; r < R; ++r) {
            if (x(c, d, t, r) < 1e-5) {
              x(c, d, t, r) = 1e-5;
            }
            obj +=
                nb_observations(c, d, t, r) * x(c, d, t, r) * durations(d, t) -
                nb_arrivals(c, d, t, r) * log(x(c, d, t, r));
            // printf("Large obj part1 %d %d %d %d: %f %f %d %f %d \t| %f\n", c,
            // d,
            //        t, r, obj, x(c, d, t, r), nb_observations(c, d, t, r),
            //        durations(d, t), nb_arrivals(c, d, t, r), obj);
            for (auto s : neighbors[r]) {
              if (alpha(r, s) != first_s_weight) {
                printf(
                    "Using alpha(%d,%d) = %f different from first alpha = %f\n",
                    r, s, alpha(r, s), first_s_weight);
                std::cin.get();
              }
              obj += 0.5 * alpha(r, s) * nb_observations(c, d, t, r) *
                     nb_observations(c, d, t, s) *
                     pow(x(c, d, t, r) - x(c, d, t, s), 2);
              // if (obj > 1e9) {
              //   printf("Neighbor part obj %d %d %d %d %d: %f %f %f %f\n", c,
              //   d,
              //          t, r, s, obj, x(c, d, t, r), x(c, d, t, s), alpha(r,
              //          s));
              // }
            }
          }
        }
      }
    }

    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          double sum = 0.0;
          for (int r = 0; r < R; ++r) {
            sum += x(c, d, t, r);
          }
          if (sum < 1e-5) {
            sum = 1e-5;
          }
          obj -= (nb_missing_arrivals(c, d, t) * log(sum));
          // if (obj > 1e9) {
          //   printf("missing data obj part2 %d %d %d: %f %f %d\n", c, d, t,
          //   obj,
          //          sum, nb_missing_arrivals(c, d, t));
          //   // std::cin.get();
          // }
        }
      }
    }

    double first_t_weight = weight[0];

    for (int c = 0; c < C; ++c) {
      for (int r = 0; r < R; ++r) {
        for (int grindex = 0; grindex < groups.size(); ++grindex) {
          auto& group = groups[grindex];
          for (auto& elem : group) {
            int d = elem.first;
            int t = elem.second;
            for (auto& elem1 : group) {
              if (elem1 == elem) {
                continue;
              }
              int d1 = elem1.first;
              int t1 = elem1.second;
              if (weight[grindex] != first_t_weight) {
                printf(
                    "Using weight[%d] = %f different from first weight = "
                    "%f\n",
                    grindex, weight[grindex], first_t_weight);
                std::cin.get();
              }
              obj += 0.5 * weight[grindex] * nb_observations(c, d, t, r) *
                     nb_observations(c, d1, t1, r) *
                     pow(x(c, d, t, r) - x(c, d1, t1, r), 2);
            }
          }
        }
      }
    }
    return obj;
  }

  xt::xarray<double> gradient(xt::xarray<double>& x) {
    xt::xarray<double> grad = xt::zeros<double>(x.shape());
    xt::xarray<double> sum = xt::zeros<double>({C, D, T});
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          double aux = 0.0;
          for (int r = 0; r < R; ++r) {
            aux += x(c, d, t, r);
          }
          sum(c, d, t) = aux;
        }
      }
    }

    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          if (sum(c, d, t) < 1e-5) {
            sum(c, d, t) = 1e-5;
          }
          double grad_component =
              nb_observations(c, d, t, 0) * durations(d, t) -
              nb_missing_arrivals(c, d, t) / sum(c, d, t);
          for (int r = 0; r < R; ++r) {
            if (x(c, d, t, r) < 1e-5) {
              x(c, d, t, r) = 1e-5;
            }
            double grad_component1 =
                grad_component - (nb_arrivals(c, d, t, r) / x(c, d, t, r));
            for (int s : neighbors[r]) {
              grad_component1 += 2 * alpha(r, s) * nb_observations(c, d, t, r) *
                                 nb_observations(c, d, t, s) *
                                 (x(c, d, t, r) - x(c, d, t, s));
            }
            grad(c, d, t, r) = grad_component1;
          }
        }
      }
    }

    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int r = 0; r < R; ++r) {
            for (auto& elem : groups[which_group(d, t)]) {
              int d1 = elem.first;
              int t1 = elem.second;
              grad(c, d, t, r) += 2 * weight[which_group(d, t)] *
                                  nb_observations(c, d, t, r) *
                                  nb_observations(c, d1, t1, r) *
                                  (x(c, d, t, r) - x(c, d1, t1, r));
            }
          }
        }
      }
    }

    return grad;
  }

  double get_rhs(xt::xarray<double>& grad, xt::xarray<double>& dir) {
    double rhs = 0.0;
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int r = 0; r < R; ++r) {
            rhs += grad(c, d, t, r) * dir(c, d, t, r);
          }
        }
      }
    }

    return rhs;
  }
  xt::xarray<double> projection(xt::xarray<double>& x) {
    xt::xarray<double> z = xt::zeros<double>(x.shape());
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int r = 0; r < R; ++r) {
            z(c, d, t, r) = std::max(param.EPS, x(c, d, t, r));
          }
        }
      }
    }
    return z;
  }
  bool is_feasible(xt::xarray<double>& x) { return true; }
  double get_lower_bound(xt::xarray<double>& x, xt::xarray<double>& grad) {
    return 0.0;
  }
};

xt::xarray<int> mnrnd(int n, const std::vector<double>& p, int S) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<int> dist(p.begin(), p.end());

  int R = p.size();
  xt::xarray<int> result = xt::zeros<int>({S, R});

  for (int s = 0; s < S; s++) {
    for (int i = 0; i < n; i++) {
      int k = dist(gen);
      result(s, k) += 1;
    }
  }

  return result;
}

class MissingLambdaCovariatesModel {
 public:
  std::string name = "Covariates";
  ulong C;
  ulong D;
  ulong T;
  ulong R;
  ulong S;

  xt::xarray<int> nb_observations;
  xt::xarray<int> nb_arrivals;
  xt::xarray<int> nb_missing_arrivals;
  xt::xarray<int> sample_arrivals;
  xt::xarray<int> sample_missing_arrivals;
  std::vector<double> pop;
  xt::xarray<double> durations;
  laspated::Param& param;

  std::vector<int> factorials_cache;

  void init_factorial_cache(int max_n) {
    factorials_cache.resize(max_n + 1);
    factorials_cache[0] = 1.0;
    for (int i = 1; i <= max_n; ++i) {
      factorials_cache[i] = factorials_cache[i - 1] * i;
    }
  }

  int factorial(int n) {
    if (n < factorials_cache.size()) {
      return factorials_cache[n];
    } else {
      int prod = factorials_cache.back();
      for (int i = factorials_cache.size(); i <= n; ++i) {
        prod *= i;
      }
      return prod;
    }
  }

  std::pair<double, double> mean_var_u(xt::xarray<int>& mn_samples, int c,
                                       int d, int t, int n,
                                       xt::xarray<double>& x) {
    double u = 0.0;
    for (int s = 0; s < S; ++s) {
      double aux = 1.0;
      for (int r = 0; r < R; ++r) {
        int power = mn_samples(s, r) + sample_arrivals(c, d, t, r, n);
        double iaux = exp(-durations(d, t) * x(c, d, t, r));
        iaux = iaux * pow(durations(d, t) * x(c, d, t, r), power);
        double denom = factorial(power);
        iaux = iaux / denom;
        aux = aux * iaux;
      }
      u += aux;
    }
    u = u / S;
    double var_u = 0.0;
    for (int s = 0; s < S; ++s) {
      double aux = 1.0;
      for (int r = 0; r < R; ++r) {
        int power = mn_samples(s, r) + sample_arrivals(c, d, t, r, n);
        double iaux = exp(-durations(d, t) * x(c, d, t, r));
        iaux = iaux * pow(durations(d, t) * x(c, d, t, r), power);
        double denom = factorial(power);
        iaux = iaux / denom;
        aux = aux * iaux;
      }
      var_u += aux * aux;
    }
    var_u = var_u / S - u * u;
    return std::make_pair(u, var_u);
  }

  std::pair<double, double> mean_var_u_optimized(xt::xarray<int>& mn_samples,
                                                 int c, int d, int t, int n,
                                                 const xt::xarray<double>& x) {
    int missing = sample_missing_arrivals(c, d, t, n);
    double lambda_r[R];
    double log_exp_r[R];
    int base_power[R];

    // Pré-cálculo dos valores que não dependem de s
    double dur = durations(d, t);
    for (int r = 0; r < R; r++) {
      lambda_r[r] = dur * x(c, d, t, r);
      log_exp_r[r] = -lambda_r[r];
      base_power[r] = sample_arrivals(c, d, t, r, n);
    }

    std::vector<double> f_values(S);

    // Computa f(s) usando log-domain
    for (int s = 0; s < S; s++) {
      double log_f = 0.0;

      for (int r = 0; r < R; r++) {
        int power = mn_samples(s, r) + base_power[r];

        // log f_r = -λ + power*log λ - log(power!)
        log_f +=
            log_exp_r[r] + power * std::log(lambda_r[r]) - lgamma(power + 1.0);
      }

      f_values[s] = std::exp(log_f);  // f(s)
    }

    // Média
    double u = 0.0;
    for (double v : f_values) u += v;
    u /= S;

    // Variância populacional
    double var_u = 0.0;
    for (double v : f_values) var_u += v * v;
    var_u = var_u / S - u * u;

    return {u, var_u};
  }

  MissingLambdaCovariatesModel(xt::xarray<int> a_nb_observations,
                               xt::xarray<int> a_nb_arrivals,
                               xt::xarray<int> a_nb_missing_arrivals,
                               xt::xarray<int> a_sample_arrivals,
                               xt::xarray<int> a_sample_missing_arrivals,
                               ulong a_S, std::vector<double>& a_pop,
                               xt::xarray<double> a_durations,
                               laspated::Param& a_param)
      : param(a_param) {
    nb_observations = a_nb_observations;
    nb_arrivals = a_nb_arrivals;
    nb_missing_arrivals = a_nb_missing_arrivals;
    sample_arrivals = a_sample_arrivals;
    sample_missing_arrivals = a_sample_missing_arrivals;
    durations = a_durations;

    C = nb_observations.shape(0);
    D = nb_observations.shape(1);
    T = nb_observations.shape(2);
    R = nb_observations.shape(3);
    S = a_S;
    pop = a_pop;
    init_factorial_cache(1000);
  }

  double f(xt::xarray<double>& x) {
    double obj = 0.0;
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int r = 0; r < R; ++r) {
            obj +=
                nb_observations(c, d, t, r) * x(c, d, t, r) * durations(d, t);
          }
        }
      }
    }

    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int n = 0; n < nb_observations(c, d, t, 0); ++n) {
            auto mn_samples =
                mnrnd(sample_missing_arrivals(c, d, t, n), pop, S);
            // double u = 0;
            // for (int s = 0; s < S; ++s) {
            //   double aux = 1;
            //   for (int r = 0; r < R; ++r) {
            //     int power = mn_samples(s, r) + sample_arrivals(c, d, t, r,
            //     n); double iaux = exp(-durations(d, t) * x(c, d, t, r)); iaux
            //     = iaux * pow(durations(d, t) * x(c, d, t, r), power); double
            //     denom = factorial(power); iaux = iaux / denom; aux = aux *
            //     iaux;
            //   }
            //   u += aux;
            // }
            // auto result_mean_var = mean_var_u(mn_samples, c, d, t, n, x);
            // double u = result_mean_var.first;
            auto result2 = mean_var_u_optimized(mn_samples, c, d, t, n, x);
            double u = result2.first;
            // if (std::abs(u - u2) > 1e-6) {
            //   printf("Difference in u: %f vs %f\n", u, u2);
            //   std::cin.get();
            // }
            obj -= log(u / S);
          }
        }
      }
    }
    return obj;
  }

  xt::xarray<double> gradient(xt::xarray<double>& x) {
    xt::xarray<double> grad = xt::zeros<double>(x.shape());
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int r = 0; r < R; ++r) {
            grad(c, d, t, r) = durations(d, t) * nb_observations(c, d, t, r);
            for (int n = 0; n < nb_observations(c, d, t, 0); ++n) {
              auto mn_samples =
                  mnrnd(sample_missing_arrivals(c, d, t, n), pop, S);
              double u = 0;
              for (int s = 0; s < S; ++s) {
                double aux = 1;
                for (int k = 0; k < R; ++k) {
                  int power = mn_samples(s, k) + sample_arrivals(c, d, t, k, n);
                  double iaux = exp(-durations(d, t) * x(c, d, t, k));
                  iaux = iaux * pow(durations(d, t) * x(c, d, t, k), power);
                  double denom = factorial(power);
                  iaux = iaux / denom;
                  aux = aux * iaux;
                }
                u += aux;
              }
              u = u / S;
              double up = 0;
              for (int s = 0; s < S; ++s) {
                int power_r = mn_samples(s, r) + sample_arrivals(c, d, t, r, n);
                double faux = exp(-x(c, d, t, r) * durations(d, t)) *
                              pow(x(c, d, t, r), power_r) *
                              (-durations(d, t) * x(c, d, t, r) + power_r);
                double iaux1 = 1;
                for (int k = 0; k < R; ++k) {
                  int power_k =
                      mn_samples(s, k) + sample_arrivals(c, d, t, k, n);
                  iaux1 = iaux1 * pow(durations(d, t), power_k);
                  double denom = factorial(power_k);
                  iaux1 = iaux1 / denom;
                }
                double iaux2 = 1;
                for (int k = 0; k < R; ++k) {
                  if (k != r) {
                    int power_k =
                        mn_samples(s, k) + sample_arrivals(c, d, t, k, n);
                    iaux2 = iaux2 * exp(-durations(d, t) * x(c, d, t, k)) *
                            pow(x(c, d, t, k), power_k);
                  }
                }
                up = up + faux * iaux1 * iaux2;
              }
              up = up / S;
              grad(c, d, t, r) -= up / u;
            }
          }
        }
      }
    }
    return grad;
  }

  xt::xarray<double> gradient_new(xt::xarray<double>& x) {
    xt::xarray<double> grad = xt::zeros<double>(x.shape());

    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          double dur = durations(d, t);

          for (int n = 0; n < nb_observations(c, d, t, 0); ++n) {
            int missing = sample_missing_arrivals(c, d, t, n);
            auto mn_samples = mnrnd(missing, pop, S);

            // Precompute arrival counts and lambdas
            int base_power[R];
            double lambda[R];
            for (int r = 0; r < R; r++) {
              base_power[r] = sample_arrivals(c, d, t, r, n);
              lambda[r] = dur * x(c, d, t, r);
            }

            std::vector<double> f(S);
            std::vector<double> f_deriv_r[R];
            for (int r = 0; r < R; r++) f_deriv_r[r].resize(S);

            // Monte Carlo
            for (int s = 0; s < S; s++) {
              double log_fs = 0.0;
              int power[R];

              for (int r = 0; r < R; r++) {
                power[r] = mn_samples(s, r) + base_power[r];
                log_fs += -lambda[r] + power[r] * std::log(lambda[r]) -
                          lgamma(power[r] + 1.0);
              }

              f[s] = std::exp(log_fs);

              // ∂ log f / ∂ λ_r
              for (int r = 0; r < R; r++) {
                double dlogf_dlambda = -1.0 + power[r] / lambda[r];
                f_deriv_r[r][s] = f[s] * dlogf_dlambda;
              }
            }

            // Mean and gradient accumulation
            double mean_f = std::accumulate(f.begin(), f.end(), 0.0) / S;

            for (int r = 0; r < R; r++) {
              double mean_df = 0.0;
              for (int s = 0; s < S; s++) mean_df += f_deriv_r[r][s];
              mean_df /= S;

              grad(c, d, t, r) -= mean_df / mean_f;
            }
          }
        }
      }
    }
    xt::xarray<double> grad_old = gradient(x);

    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int r = 0; r < R; ++r) {
            if (std::abs(grad(c, d, t, r) - grad_old(c, d, t, r)) > 1e-6) {
              printf("Difference in gradient at %d %d %d %d: %f vs %f\n", c, d,
                     t, r, grad(c, d, t, r), grad_old(c, d, t, r));
              std::cin.get();
            }
          }
        }
      }
    }

    return grad;
  }

  double get_rhs(xt::xarray<double>& grad, xt::xarray<double>& dir) {
    double rhs = 0.0;
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int r = 0; r < R; ++r) {
            rhs += grad(c, d, t, r) * dir(c, d, t, r);
          }
        }
      }
    }

    return rhs;
  }
  xt::xarray<double> projection(xt::xarray<double>& x) {
    xt::xarray<double> z = xt::zeros<double>(x.shape());
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int r = 0; r < R; ++r) {
            z(c, d, t, r) = std::max(param.EPS, x(c, d, t, r));
          }
        }
      }
    }
    return z;
  }
  bool is_feasible(xt::xarray<double>& x) { return true; }
  double get_lower_bound(xt::xarray<double>& x, xt::xarray<double>& grad) {
    return 0.0;
  }
};

class MissingLambdaArrivalCovariatesModel {
 public:
  GRBEnv env;
  laspated::Param& param;
  std::string name = "Covariates";
  ulong C;
  ulong D;
  ulong T;
  ulong R;
  ulong nb_regressors;

  xt::xarray<int> nb_observations;
  xt::xarray<int> nb_arrivals;
  xt::xarray<int> nb_missing_arrivals;
  xt::xarray<double> regressors;
  xt::xarray<double> durations;

  MissingLambdaArrivalCovariatesModel(xt::xarray<int>& a_nb_observations,
                                      xt::xarray<int>& a_nb_arrivals,
                                      xt::xarray<int>& a_nb_missing_arrivals,
                                      xt::xarray<double>& a_durations,
                                      xt::xarray<double>& reg,
                                      laspated::Param& param)
      : param(param) {
    C = a_nb_observations.shape(0);
    D = a_nb_observations.shape(1);
    T = a_nb_observations.shape(2);
    R = a_nb_observations.shape(3);
    durations = a_durations;
    nb_regressors = reg.shape(0);
    nb_observations = a_nb_observations;
    nb_arrivals = a_nb_arrivals;
    nb_missing_arrivals = a_nb_missing_arrivals;
    regressors = reg;
  }

  double f(xt::xarray<double>& x) {
    double obj = 0.0;
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          double sum_mu = 0.0;
          double sum_y_log_mu = 0.0;
          for (int r = 0; r < R; ++r) {
            // Model assumption:
            // mu_cdtr = lambda_cdtr * durations(d,t) = <x(c,d,t,:), reg(:,r)>.
            double mu = 0.0;
            for (int j = 0; j < nb_regressors; ++j) {
              mu += x(c, d, t, j) * regressors(j, r);
            }
            mu = std::max(mu, param.EPS);
            sum_mu += mu;
            sum_y_log_mu += nb_arrivals(c, d, t, r) * log(mu);
          }
          sum_mu = std::max(sum_mu, param.EPS);
          obj += nb_observations(c, d, t, 0) * sum_mu - sum_y_log_mu -
                 nb_missing_arrivals(c, d, t) * log(sum_mu);
        }
      }
    }

    return obj;
  }

  xt::xarray<double> gradient(xt::xarray<double>& x) {
    xt::xarray<double> grad = xt::zeros<double>(x.shape());
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          double sum_mu = 0.0;
          for (int r = 0; r < R; ++r) {
            double mu = 0.0;
            for (int j = 0; j < nb_regressors; ++j) {
              mu += x(c, d, t, j) * regressors(j, r);
            }
            mu = std::max(mu, param.EPS);
            sum_mu += mu;
          }
          sum_mu = std::max(sum_mu, param.EPS);
          for (int j = 0; j < nb_regressors; ++j) {
            double sum_reg_j = 0.0;
            for (int r = 0; r < R; ++r) {
              sum_reg_j += regressors(j, r);
            }
            double grad_component = nb_observations(c, d, t, 0) * sum_reg_j;
            for (int r = 0; r < R; ++r) {
              double mu = 0.0;
              for (int k = 0; k < nb_regressors; ++k) {
                mu += x(c, d, t, k) * regressors(k, r);
              }
              mu = std::max(mu, param.EPS);
              grad_component -= nb_arrivals(c, d, t, r) * regressors(j, r) / mu;
            }
            grad_component -= nb_missing_arrivals(c, d, t) * sum_reg_j / sum_mu;
            grad(c, d, t, j) = grad_component;
          }
        }
      }
    }
    return grad;
  }

  double get_rhs(xt::xarray<double>& grad, xt::xarray<double>& dir) {
    double rhs = 0.0;
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int j = 0; j < nb_regressors; ++j) {
            rhs += grad(c, d, t, j) * dir(c, d, t, j);
          }
        }
      }
    }

    return rhs;
  }

  xt::xarray<double> projection(xt::xarray<double>& x) {
    using namespace std;
    xt::xarray<GRBVar> y({C, D, T, nb_regressors});
    GRBModel model(env);
    stringstream name;

    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int j = 0; j < nb_regressors; ++j) {
            name << "y_" << c << "_" << d << "_" << t << "_" << j;
            double ub = (j == 0) ? 1 : param.upper_lambda;
            y(c, d, t, j) = model.addVar(0, ub, 0, GRB_CONTINUOUS, name.str());
            name.str("");
          }
        }
      }
    }

    GRBQuadExpr obj = 0;
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int j = 0; j < nb_regressors; ++j) {
            obj += 0.5 * y(c, d, t, j) * y(c, d, t, j) -
                   x(c, d, t, j) * y(c, d, t, j);
          }
        }
      }
    }
    // cin.get();
    try {
      model.setObjective(obj, GRB_MINIMIZE);
    } catch (GRBException& ex) {
      cout << ex.getMessage() << "\n";
      exit(1);
    }

    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          for (int r = 0; r < R; ++r) {
            GRBLinExpr con1 = 0;
            for (int j = 0; j < nb_regressors; ++j) {
              con1 += y(c, d, t, j) * regressors(j, r);
            }
            name << "con1_" << c << "_" << d << "_" << t << "_" << r;
            model.addConstr(con1, GRB_GREATER_EQUAL, param.EPS, name.str());
            name.str("");
            con1 = 0;
          }
        }
      }
    }

    model.update();
    model.set(GRB_IntParam_OutputFlag, 0);
    model.set(GRB_IntParam_NumericFocus, 3);
    model.set(GRB_IntParam_DualReductions, 0);
    model.optimize();

    auto status = model.get(GRB_IntAttr_Status);

    xt::xarray<double> y_val = GRB_INFINITY * xt::ones<double>(y.shape());
    if (status == GRB_OPTIMAL) {
      for (int c = 0; c < C; ++c) {
        double sum_c = 0.0;
        for (int d = 0; d < D; ++d) {
          for (int t = 0; t < T; ++t) {
            for (int j = 0; j < nb_regressors; ++j) {
              y_val(c, d, t, j) = y(c, d, t, j).get(GRB_DoubleAttr_X);
              for (int r = 0; r < R; ++r) {
                sum_c += y_val(c, d, t, j) * regressors(j, r);
              }
            }
          }
        }
      }
    } else {
      cout << "Error. Projection problem solved with status = " << status
           << "\n";
      model.write("projection_model.lp");
      cout << "Wrote model at projection_model.lp\n";
      exit(status);
    }
    return y_val;
  }

  bool is_feasible(xt::xarray<double>& x) { return true; }

  double get_lower_bound(xt::xarray<double>& x, xt::xarray<double>& grad) {
    return 0.0;
  }
};

laspated::CrossValidationResult cross_validation(
    laspated::Param& param, MissingLambdaRegularizedModel& model,
    xt::xarray<int>& sample, xt::xarray<int>& sample_missing_arrivals,
    std::vector<double>& group_weights, const std::vector<int>& daily_obs) {
  if (daily_obs.size() != model.D) {
    throw std::invalid_argument(
        "daily_obs size must match model.D in cross_validation.");
  }
  std::vector<double> alphas = group_weights;
  auto t0 = std::chrono::high_resolution_clock::now();
  int nb_observations_total = sample.shape(4);
  int nb_groups = model.groups.size();
  double min_loss = GRB_INFINITY;
  double wall_time = 0;

  int nb_in_block = floor(nb_observations_total * param.cv_proportion);
  int nb_folds = floor(1.0 / param.cv_proportion);
  xt::xarray<int> initial_nb_obs = model.nb_observations;
  xt::xarray<int> initial_nb_arrivals = model.nb_arrivals;
  xt::xarray<int> initial_nb_missing_arrivals = model.nb_missing_arrivals;

  double best_alpha = GRB_INFINITY;
  double best_weight = GRB_INFINITY;
  // printf("cv_proportion = %f, nb_in_block = %d, nb_observations_total =
  // %d\n",
  //        param.cv_proportion, nb_in_block, nb_observations_total);
  for (int index_alpha = 0; index_alpha < alphas.size(); ++index_alpha) {
    // printf("alpha = %f, nb_in_block = %d\n", alphas[index_alpha],
    // nb_in_block);
    double likelihood = 0;
    model.alpha = alphas[index_alpha] * xt::ones<double>({model.R, model.R});
    model.weight =
        std::vector<double>(model.groups.size(), group_weights[index_alpha]);
    for (int index_cross = 0; index_cross < nb_folds; ++index_cross) {
      xt::xarray<int> nb_observations_current =
          xt::zeros<int>({model.C, model.D, model.T, model.R});
      xt::xarray<int> nb_arrivals_current =
          xt::zeros<int>({model.C, model.D, model.T, model.R});
      xt::xarray<int> nb_missing_arrivals_current =
          xt::zeros<int>({model.C, model.D, model.T});
      for (int index = index_cross * nb_in_block;
           index < (index_cross + 1) * nb_in_block; ++index) {
        for (int c = 0; c < model.C; ++c) {
          for (int d = 0; d < model.D; ++d) {
            if (index >= daily_obs[d]) {
              continue;
            }
            for (int t = 0; t < model.T; ++t) {
              nb_missing_arrivals_current(c, d, t) +=
                  sample_missing_arrivals(c, d, t, index);
              for (int r = 0; r < model.R; ++r) {
                ++nb_observations_current(c, d, t, r);
                nb_arrivals_current(c, d, t, r) += sample(c, d, t, r, index);
              }
            }
          }
        }
      }
      xt::xarray<double> x0 =
          param.EPS * xt::ones<double>({model.C, model.D, model.T, model.R});
      model.nb_observations = nb_observations_current;
      model.nb_arrivals = nb_arrivals_current;
      model.nb_missing_arrivals = nb_missing_arrivals_current;
      param.max_iter = 100;
      xt::xarray<double> x = laspated::projected_gradient_armijo_feasible<
          MissingLambdaRegularizedModel>(model, param, x0);
      xt::xarray<int> nb_arrivals_remaining =
          xt::zeros<int>({model.C, model.D, model.T, model.R});
      xt::xarray<int> nb_missing_arrivals_remaining =
          xt::zeros<int>({model.C, model.D, model.T});
      std::vector<int> nb_remaining_by_day(model.D, 0);
      for (int d = 0; d < model.D; ++d) {
        int day_obs = daily_obs[d];
        int val_start = index_cross * nb_in_block;
        int val_end = (index_cross + 1) * nb_in_block;
        int start_clip = std::min(day_obs, val_start);
        int end_clip = std::min(day_obs, val_end);
        int heldout = std::max(0, end_clip - start_clip);
        nb_remaining_by_day[d] = day_obs - heldout;
      }
      for (int index = 0; index < index_cross * nb_in_block; ++index) {
        for (int c = 0; c < model.C; ++c) {
          for (int d = 0; d < model.D; ++d) {
            if (index >= daily_obs[d]) {
              continue;
            }
            for (int t = 0; t < model.T; ++t) {
              nb_missing_arrivals_remaining(c, d, t) +=
                  sample_missing_arrivals(c, d, t, index);
              for (int r = 0; r < model.R; ++r) {
                nb_arrivals_remaining(c, d, t, r) += sample(c, d, t, r, index);
              }
            }
          }
        }
      }
      for (int index = (index_cross + 1) * nb_in_block;
           index < nb_observations_total; ++index) {
        for (int c = 0; c < model.C; ++c) {
          for (int d = 0; d < model.D; ++d) {
            if (index >= daily_obs[d]) {
              continue;
            }
            for (int t = 0; t < model.T; ++t) {
              nb_missing_arrivals_remaining(c, d, t) +=
                  sample_missing_arrivals(c, d, t, index);
              for (int r = 0; r < model.R; ++r) {
                nb_arrivals_remaining(c, d, t, r) += sample(c, d, t, r, index);
              }
            }
          }
        }
      }
      double f = 0.0;
      for (int c = 0; c < model.C; ++c) {
        for (int d = 0; d < model.D; ++d) {
          for (int t = 0; t < model.T; ++t) {
            double S = 0.0;
            for (int r = 0; r < model.R; ++r) {
              // compute likelihood for Missing Lambda Model
              int this_nb_obs = nb_remaining_by_day[d];
              f += this_nb_obs * x(c, d, t, r) * model.durations(d, t) -
                   nb_arrivals_remaining(c, d, t, r) * log(x(c, d, t, r));
              S += x(c, d, t, r);
            }
            f -= nb_missing_arrivals_remaining(c, d, t) * log(S);
          }
        }
      }
      likelihood += f;
      // printf("\t\tf = %f\n", f);
    }
    // printf("\tlikelihood = %f\n", likelihood);
    likelihood = likelihood / nb_folds;
    if (likelihood < min_loss) {
      min_loss = likelihood;
      best_alpha = alphas[index_alpha];
      best_weight = group_weights[index_alpha];
    }
  }
  model.alpha = best_alpha * xt::ones<double>({model.R, model.R});
  model.weight = std::vector<double>(model.groups.size(), best_weight);
  xt::xarray<int> nb_observations_current =
      xt::zeros<int>({model.C, model.D, model.T, model.R});
  xt::xarray<int> nb_arrivals_current =
      xt::zeros<int>({model.C, model.D, model.T, model.R});
  xt::xarray<int> nb_missing_arrivals_current =
      xt::zeros<int>({model.C, model.D, model.T});
  for (int c = 0; c < model.C; ++c) {
    for (int d = 0; d < model.D; ++d) {
      for (int t = 0; t < model.T; ++t) {
        for (int r = 0; r < model.R; ++r) {
          for (int index = 0; index < daily_obs[d]; ++index) {
            ++nb_observations_current(c, d, t, r);
            nb_arrivals_current(c, d, t, r) += sample(c, d, t, r, index);
            nb_missing_arrivals_current(c, d, t) +=
                sample_missing_arrivals(c, d, t, index);
          }
        }
      }
    }
  }
  model.nb_observations = nb_observations_current;
  model.nb_arrivals = nb_arrivals_current;
  model.nb_missing_arrivals = nb_missing_arrivals_current;
  xt::xarray<double> x0 =
      param.EPS * xt::ones<double>({model.C, model.D, model.T, model.R});
  xt::xarray<double> x = laspated::projected_gradient_armijo_feasible<
      MissingLambdaRegularizedModel>(model, param, x0);
  auto dt = std::chrono::high_resolution_clock::now() - t0;
  wall_time = std::chrono::duration_cast<std::chrono::seconds>(dt).count();
  model.nb_observations = initial_nb_obs;
  model.nb_arrivals = initial_nb_arrivals;
  model.nb_missing_arrivals = initial_nb_missing_arrivals;
  model.alpha = best_alpha * xt::ones<double>({model.R, model.R});
  model.weight = std::vector<double>(model.groups.size(), best_weight);
  x0 = param.EPS * xt::ones<double>({model.C, model.D, model.T, model.R});
  param.max_iter = 100;
  x = laspated::projected_gradient_armijo_feasible<
      MissingLambdaRegularizedModel>(model, param, x0);
  std::stringstream out_name;
  out_name << "results/lambda_model2_cv_R" << model.R << "_T" << model.T
           << ".txt";
  std::ofstream arq(out_name.str(), std::ios::out);
  for (int c = 0; c < model.C; ++c) {
    for (int d = 0; d < model.D; ++d) {
      for (int t = 0; t < model.T; ++t) {
        int ind_t = d * model.T + t;
        for (int r = 0; r < model.R; ++r) {
          arq << c << " " << r << " " << ind_t << " " << x(c, d, t, r) << "\n";
        }
      }
    }
  }
  arq.close();
  printf("wall_time = %f seconds, best_weight = %f\n", wall_time, best_weight);
  return {wall_time, best_weight, x};
}

xt::xarray<double> MSE(xt::xarray<double>& x, xt::xarray<double>& y) {
  if (x.shape() != y.shape()) {
    throw std::invalid_argument(
        "Shapes of x and y must be the same for MSE calculation.");
  }

  return xt::square(x - y);
}
