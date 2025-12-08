#include "missing_data.hpp"

typedef struct {
  double prob;
  xt::xarray<double> probs;
  xt::xarray<double> lambda_missing;
  xt::xarray<double> lambda_no_missing;
} ResultModel1;

class AppParams {
 public:
  double EPS;           // Epsilon for feasibility/convergence
  double sigma;         // Sigma for armijo step
  int max_iter;         // Max number of iterations.
  double lower_lambda;  // lower bound on decision variables
  double beta_bar;      // Initial step size in projected gradient
  ulong S;              // Number of Monte Carlo samples

  std::vector<double> test_weights;
  std::string model;
  std::string info_file;
  std::string arrivals_file;
  std::string neighbors_file;
  std::string missing_file;
  std::string mn_samples_file;
};

xt::xarray<double> get_confidence_intervals(
    xt::xarray<int>& nb_observations, xt::xarray<int>& nb_arrivals,
    xt::xarray<int>& nb_missing_arrivals, xt::xarray<double>& durations,
    ResultModel1& result) {
  ulong C = nb_observations.shape(0);
  ulong D = nb_observations.shape(1);
  ulong T = nb_observations.shape(2);
  ulong R = nb_observations.shape(3);
  // for (ulong c = 0; c < C; ++c) {
  //   for (ulong d = 0; d < D; ++d) {
  //     for (ulong t = 0; t < T; ++t) {
  //       double p = result.probs(c, d, t);
  //       for (ulong r = 0; r < R; ++r) {
  //         double N = nb_observations(c, d, t, r);
  //         double D_t = durations(d, t);
  //         double lambda = result.lambda_missing(c, d, t, r);

  //         double var_lambda = lambda / ((1.0 - p) * N * D_t);
  //         double se_lambda = std::sqrt(var_lambda);
  //         conf_intervals(c, d, t, r, 0) = std::max(0.0, lambda - z *
  //         se_lambda); conf_intervals(c, d, t, r, 1) = lambda + z * se_lambda;
  //       }
  //     }
  //   }
  // }
  xt::xarray<double> conf_intervals = xt::zeros<double>({C, D, T, R, (ulong)2});
  for (ulong c = 0; c < C; ++c) {
    for (ulong d = 0; d < D; ++d) {
      for (ulong t = 0; t < T; ++t) {
        double S = 0.0;
        double p = result.probs(c, d, t);
        for (ulong r = 0; r < R; ++r) {
          S += result.lambda_missing(c, d, t, r);
        }
        std::vector<ulong> nonzero_lambdas;
        for (ulong r = 0; r < R; ++r) {
          if (result.lambda_missing(c, d, t, r) > 0.0001) {
            nonzero_lambdas.push_back(r);
          }
        }
        // printf("nonzero_lambdas.size() = %d\n", nonzero_lambdas.size());
        xt::xarray<double> fisher =
            xt::zeros<double>({nonzero_lambdas.size(), nonzero_lambdas.size()});
        for (ulong r1 = 0; r1 < nonzero_lambdas.size(); ++r1) {
          double lam = result.lambda_missing(c, d, t, nonzero_lambdas[r1]);
          if (lam < 0.01) {
            lam = 0.01;
          }
          fisher(r1, r1) =
              p * durations(d, t) / S + ((1 - p) * durations(d, t)) / lam;
          for (ulong r2 = 0; r2 < nonzero_lambdas.size(); ++r2) {
            if (r1 != r2) {
              fisher(r1, r2) = p * durations(d, t) / S;
            }
          }
        }
        auto eig = xt::linalg::eigvals(fisher);
        auto abs_eig = xt::abs(eig);
        double max_eig = xt::amax(abs_eig)();
        double eps = 1e-3;
        double delta = max_eig * eps;
        auto fisher_reg =
            fisher + delta * xt::eye<double>(nonzero_lambdas.size());
        xt::xarray<double> inv_fisher = xt::linalg::inv(fisher);
        auto eig_reg = xt::linalg::eigvals(fisher_reg);
        auto abs_eig_reg = xt::abs(eig);
        double cond_number = xt::amax(abs_eig)() / xt::amin(abs_eig)();
        // std::cout << "Cond. number reg: " << cond_number
        //           << " max = " << xt::amax(abs_eig)()
        //           << " min = " << xt::amin(abs_eig)() << "\n";
        // std::cin.get();
        xt::xarray<double> prod = xt::linalg::dot(fisher, inv_fisher);
        for (ulong i = 0; i < nonzero_lambdas.size(); ++i) {
          for (ulong j = 0; j < nonzero_lambdas.size(); ++j) {
            if (i == j) {
              assert(std::abs(prod(i, j) - 1.0) < 1e-8);
            } else {
              assert(std::abs(prod(i, j)) < 1e-8);
            }
          }
        }

        double phi_inv = 1.96;
        for (ulong r = 0; r < nonzero_lambdas.size(); ++r) {
          double std_lambda = sqrt(inv_fisher(r, r));
          conf_intervals(c, d, t, nonzero_lambdas[r], 0) =
              result.lambda_missing(c, d, t, nonzero_lambdas[r]) -
              phi_inv * std_lambda;
          if (conf_intervals(c, d, t, nonzero_lambdas[r], 0) < 0.0) {
            conf_intervals(c, d, t, nonzero_lambdas[r], 0) = 0.0;
          }
          conf_intervals(c, d, t, nonzero_lambdas[r], 1) =
              result.lambda_missing(c, d, t, nonzero_lambdas[r]) +
              phi_inv * std_lambda;
        }
      }
    }
  }

  return conf_intervals;
}

namespace po = boost::program_options;
ResultModel1 run_model1(xt::xarray<int>& nb_observations,
                        xt::xarray<int>& nb_arrivals,
                        xt::xarray<int>& nb_missing_arrivals,
                        xt::xarray<double>& durations) {
  ulong C = nb_observations.shape(0);
  ulong D = nb_observations.shape(1);
  ulong T = nb_observations.shape(2);
  ulong R = nb_observations.shape(3);

  double without_location = 0;
  double with_location = 0;
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      for (int t = 0; t < T; ++t) {
        without_location += nb_missing_arrivals(c, d, t);
        for (int r = 0; r < R; ++r) {
          with_location += nb_arrivals(c, d, t, r);
        }
      }
    }
  }

  double prob = without_location / (without_location + with_location);
  xt::xarray<int> no_location = xt::zeros<int>({C, D, T});
  xt::xarray<int> yes_location = xt::zeros<int>({C, D, T});
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      for (int t = 0; t < T; ++t) {
        no_location(c, d, t) += nb_missing_arrivals(c, d, t);
        for (int r = 0; r < R; ++r) {
          yes_location(c, d, t) += nb_arrivals(c, d, t, r);
        }
      }
    }
  }

  xt::xarray<double> probs = xt::zeros<double>({C, D, T});
  xt::xarray<double> lambdas_missing = xt::zeros<double>({C, D, T, R});
  xt::xarray<double> lambdas_no_missing = xt::zeros<double>({C, D, T, R});

  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      for (int t = 0; t < T; ++t) {
        probs(c, d, t) = static_cast<double>(no_location(c, d, t)) /
                         (no_location(c, d, t) + yes_location(c, d, t));
      }
    }
  }

  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      for (int t = 0; t < T; ++t) {
        for (int r = 0; r < R; ++r) {
          lambdas_no_missing(c, d, t, r) =
              nb_arrivals(c, d, t, r) /
              (nb_observations(c, d, t, r) * durations(d, t));
          lambdas_missing(c, d, t, r) =
              nb_arrivals(c, d, t, r) /
              ((1.0 - probs(c, d, t)) * nb_observations(c, d, t, r) *
               durations(d, t));
        }
      }
    }
  }

  auto result = ResultModel1{prob, probs, lambdas_missing, lambdas_no_missing};

  std::stringstream out_filename;
  out_filename << "results/lambda_model1_R" << R << "_T" << T << ".txt";

  std::ofstream model1_lambda_file(out_filename.str(), std::ios::out);
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      for (int t = 0; t < T; ++t) {
        int ind_t = d * T + t;
        for (int r = 0; r < R; ++r) {
          model1_lambda_file << c << " " << r << " " << ind_t << " "
                             << result.lambda_no_missing(c, d, t, r) << " "
                             << result.lambda_missing(c, d, t, r) << "\n";
        }
      }
    }
  }
  model1_lambda_file.close();

  out_filename.str("");
  out_filename << "results/p_model1_R" << R << "_T" << T << ".txt";

  std::ofstream model1_p_file(out_filename.str(), std::ios::out);
  model1_p_file << result.prob << "\n";
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      for (int t = 0; t < T; ++t) {
        int ind_t = d * T + t;
        model1_p_file << c << " " << ind_t << " " << result.probs(c, d, t)
                      << "\n";
      }
    }
  }
  model1_p_file.close();
  // xt::xarray<double> confidence_intervals = get_confidence_intervals(
  //     nb_observations, nb_arrivals, nb_missing_arrivals, durations, result);

  // // TODO: 3 figures, one for each c \in C
  // //  conf_0, lambda, conf_1
  // std::ofstream model1_conf_intervals("results/conf_intervals_model1.txt",
  //                                     std::ios::out);
  // for (int c = 0; c < C; ++c) {
  //   for (int d = 0; d < D; ++d) {
  //     for (int t = 0; t < T; ++t) {
  //       int ind_t = d * T + t;
  //       for (int r = 0; r < R; ++r) {
  //         model1_conf_intervals << c << " " << r << " " << ind_t << " "
  //                               << confidence_intervals(c, d, t, r, 0) << " "
  //                               << result.lambda_missing(c, d, t, r) << " "
  //                               << confidence_intervals(c, d, t, r, 1) <<
  //                               "\n";
  //       }
  //     }
  //   }
  // }
  // model1_conf_intervals.close();

  return result;
}

void set_groups(std::vector<std::vector<std::pair<int, int>>>& groups, ulong D,
                ulong T) {
  if (T == 48) {
    // Weekday morning
    for (int d = 0; d < 5; ++d) {
      for (int t = 12; t < 20; ++t) {
        groups[0].push_back(std::make_pair(d, t));
      }
    }

    // Weekday afternoon
    for (int d = 0; d < 5; ++d) {
      for (int t = 20; t < 36; ++t) {
        groups[1].push_back(std::make_pair(d, t));
      }
    }

    // Weekday evening
    for (int d = 0; d < 5; ++d) {
      for (int t = 36; t < 44; ++t) {
        groups[2].push_back(std::make_pair(d, t));
      }
    }

    // Weekday Night
    for (int d = 0; d < 4; ++d) {
      for (int t = 44; t < 48; ++t) {
        groups[3].push_back(std::make_pair(d, t));
      }
    }

    // Sunday Night
    for (int t = 44; t < 48; ++t) {
      groups[3].push_back(std::make_pair(6, t));
    }

    // Weekday Early Morning
    for (int d = 0; d < 5; ++d) {
      for (int t = 0; t < 12; ++t) {
        groups[3].push_back(std::make_pair(d, t));
      }
    }

    // Friday/Saturday Night
    for (int t = 44; t < 48; ++t) {
      groups[4].push_back(std::make_pair(4, t));
      groups[4].push_back(std::make_pair(5, t));
    }

    // Weekend Early Morning
    for (int t = 0; t < 12; ++t) {
      groups[4].push_back(std::make_pair(5, t));
      groups[4].push_back(std::make_pair(6, t));
    }

    // Weekend Morning
    for (int t = 12; t < 20; ++t) {
      groups[5].push_back(std::make_pair(5, t));
      groups[5].push_back(std::make_pair(6, t));
    }

    // Weekend Afternoon
    for (int t = 20; t < 36; ++t) {
      groups[6].push_back(std::make_pair(5, t));
      groups[6].push_back(std::make_pair(6, t));
    }
    // Weekend Evening
    for (int t = 36; t < 44; ++t) {
      groups[7].push_back(std::make_pair(5, t));
      groups[7].push_back(std::make_pair(6, t));
    }
  } else if (T == 24) {
    // Weekday morning
    for (int d = 0; d < 5; ++d) {
      for (int t = 6; t < 10; ++t) {
        groups[0].push_back(std::make_pair(d, t));
      }
    }

    // Weekday afternoon
    for (int d = 0; d < 5; ++d) {
      for (int t = 10; t < 18; ++t) {
        groups[1].push_back(std::make_pair(d, t));
      }
    }

    // Weekday evening
    for (int d = 0; d < 5; ++d) {
      for (int t = 18; t < 22; ++t) {
        groups[2].push_back(std::make_pair(d, t));
      }
    }

    // Weekday Night
    for (int d = 0; d < 4; ++d) {
      for (int t = 22; t < 24; ++t) {
        groups[3].push_back(std::make_pair(d, t));
      }
    }

    // Sunday Night
    for (int t = 22; t < 24; ++t) {
      groups[3].push_back(std::make_pair(6, t));
    }

    // Weekday Early Morning
    for (int d = 0; d < 5; ++d) {
      for (int t = 0; t < 6; ++t) {
        groups[3].push_back(std::make_pair(d, t));
      }
    }

    // Friday/Saturday Night
    for (int t = 22; t < 24; ++t) {
      groups[4].push_back(std::make_pair(4, t));
      groups[4].push_back(std::make_pair(5, t));
    }

    // Weekend Early Morning
    for (int t = 0; t < 6; ++t) {
      groups[4].push_back(std::make_pair(5, t));
      groups[4].push_back(std::make_pair(6, t));
    }

    // Weekend Morning
    for (int t = 6; t < 10; ++t) {
      groups[5].push_back(std::make_pair(5, t));
      groups[5].push_back(std::make_pair(6, t));
    }

    // Weekend Afternoon
    for (int t = 10; t < 18; ++t) {
      groups[6].push_back(std::make_pair(5, t));
      groups[6].push_back(std::make_pair(6, t));
    }
    // Weekend Evening
    for (int t = 18; t < 22; ++t) {
      groups[7].push_back(std::make_pair(5, t));
      groups[7].push_back(std::make_pair(6, t));
    }
  }
}

void run_regularized_model(
    AppParams& app_params, xt::xarray<int>& nb_observations,
    xt::xarray<int>& nb_arrivals, xt::xarray<int>& nb_missing_arrivals,
    xt::xarray<int>& samples, xt::xarray<int>& samples_missing_arrivals,
    xt::xarray<double>& durations, std::vector<std::vector<int>>& neighbors) {
  int nb_groups = 8;
  ulong C = nb_observations.shape(0);
  ulong D = nb_observations.shape(1);
  ulong T = nb_observations.shape(2);
  ulong R = nb_observations.shape(3);
  std::vector<std::vector<std::pair<int, int>>> groups(
      nb_groups, std::vector<std::pair<int, int>>());

  std::vector<double> test_weights = app_params.test_weights;
  laspated::Param param;
  param.EPS = app_params.EPS;
  param.sigma = app_params.sigma;
  param.lower_lambda = app_params.lower_lambda;
  param.max_iter = app_params.max_iter;
  param.beta_bar = app_params.beta_bar;
  param.max_iter = 100;
  std::stringstream filename;
  double sum_run_times = 0.0;
  for (double w : test_weights) {
    std::vector<double> weights(nb_groups, w);
    xt::xarray<double> alphas = w * xt::ones<double>({R, R});
    printf("Running w = %f\n", w);
    MissingLambdaRegularizedModel m1(nb_observations, nb_arrivals,
                                     nb_missing_arrivals, alphas, weights,
                                     groups, neighbors, durations, param);
    xt::xarray<double> x0 = param.EPS * xt::ones<double>({C, D, T, R});
    auto t0 = std::chrono::high_resolution_clock::now();
    xt::xarray<double> lambda = laspated::projected_gradient_armijo_feasible<
        MissingLambdaRegularizedModel>(m1, param, x0);
    auto dt = std::chrono::high_resolution_clock::now() - t0;
    sum_run_times +=
        std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
    filename.str("");
    filename << "results/lambda_model2_w" << w << "_R" << R << "_T" << T
             << ".txt";
    std::ofstream lambda_reg_file(filename.str(), std::ios::out);
    for (int c = 0; c < C; ++c) {
      for (int d = 0; d < D; ++d) {
        for (int t = 0; t < T; ++t) {
          int ind_t = d * T + t;
          for (int r = 0; r < R; ++r) {
            lambda_reg_file << c << " " << r << " " << ind_t << " "
                            << lambda(c, d, t, r) << "\n";
          }
        }
      }
    }
    std::cout << "Wrote file " << filename.str() << "\n";
    lambda_reg_file.close();
  }
  double avg_run_time = sum_run_times / test_weights.size();
  std::cout << "Average run time (ms): " << avg_run_time << "\n";

  for (size_t i = 0; i < test_weights.size(); ++i) {
    std::cout << "Test weight[" << i << "] = " << test_weights[i] << "\n";
  }

  std::vector<double> weights(nb_groups, 0.0);
  xt::xarray<double> alphas = 0.0 * xt::ones<double>({R, R});
  MissingLambdaRegularizedModel m1(nb_observations, nb_arrivals,
                                   nb_missing_arrivals, alphas, weights, groups,
                                   neighbors, durations, param);
  xt::xarray<double> x0 = param.EPS * xt::ones<double>({C, D, T, R});

  // std::vector<double> cv_weights = app_params.test_weights;
  std::vector<double> cv_weights;
  cv_weights.push_back(0.000001);
  cv_weights.push_back(0.000004);
  cv_weights.push_back(0.000006);
  cv_weights.push_back(0.000009);
  cv_weights.push_back(0.00001);

  std::sort(cv_weights.begin(), cv_weights.end());
  for (size_t i = 0; i < cv_weights.size(); ++i) {
    std::cout << "cv_weight[" << i << "] = " << cv_weights[i] << "\n";
  }
  laspated::CrossValidationResult result = cross_validation(
      param, m1, samples, samples_missing_arrivals, cv_weights);

  double best_w = result.weight;
  std::cout << "Best weight = " << best_w << "\n";
  std::cout << "Run time cv = " << result.wall_time << " s\n";
}

void run_model_population(AppParams& app_params,
                          xt::xarray<int>& nb_observations,
                          xt::xarray<int>& nb_arrivals,
                          xt::xarray<int>& nb_missing_arrivals,
                          xt::xarray<double>& durations,
                          xt::xarray<int>& sample,
                          xt::xarray<int>& sample_missing_arrivals,
                          std::vector<double>& pop, ResultModel1& result) {
  ulong C = nb_observations.shape(0);
  ulong D = nb_observations.shape(1);
  ulong T = nb_observations.shape(2);
  ulong R = nb_observations.shape(3);
  ulong nb_obs = sample.shape(4);

  ulong S = app_params.S;
  // xt::xarray<int> mn_samples = xt::zeros<int>({C, D, T, nb_obs, S, R});

  // std::ifstream mn_samples_file("Rect10x10/mn_samples.dat", std::ios::in);
  // int count = 0;
  // std::string aux_str;
  // while (true) {
  //   std::getline(mn_samples_file, aux_str);
  //   if (aux_str == "END") {
  //     break;
  //   }
  //   // std::cout << aux_str << "\n";
  //   std::istringstream ss(aux_str);
  //   int c, d, t, n, s, r, val;
  //   ss >> c >> d >> t >> n >> s >> r >> val;
  //   mn_samples(c, d, t, n, s, r) = val;
  //   ++count;
  //   if (count % (39836160) == 0) {
  //     printf("Reading mn_samples\n");
  //   }
  // }
  // mn_samples_file.close();
  // printf("Read mn_samples.txt");

  laspated::Param param;
  param.EPS = app_params.EPS;
  param.sigma = app_params.sigma;
  param.lower_lambda = app_params.lower_lambda;
  param.max_iter = app_params.max_iter;
  param.beta_bar = app_params.beta_bar;

  std::stringstream filename;

  MissingLambdaCovariatesModel m2(
      nb_observations, nb_arrivals, nb_missing_arrivals, sample,
      sample_missing_arrivals, S, pop, durations, param);

  xt::xarray<double> x0 = param.EPS * xt::ones<double>({C, D, T, R});

  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      for (int t = 0; t < T; ++t) {
        int ind_t = d * T + t;
        for (int r = 0; r < R; ++r) {
          x0(c, d, t, r) = result.lambda_missing(c, d, t, r);
        }
      }
    }
  }
  std::cout << "Beginning population model\n";
  auto t0 = std::chrono::high_resolution_clock::now();
  xt::xarray<double> lambda = laspated::projected_gradient_armijo_feasible<
      MissingLambdaCovariatesModel>(m2, param, x0);
  auto dt = std::chrono::high_resolution_clock::now() - t0;
  double run_time =
      std::chrono::duration_cast<std::chrono::milliseconds>(dt).count();
  std::cout << "Run time model with population covariate (ms): " << run_time
            << "\n";
  std::stringstream out_filename;
  out_filename << "results/lambda_model3_R" << R << "_T" << T << ".txt";
  std::ofstream lambda_model2_file(out_filename.str(), std::ios::out);
  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      for (int t = 0; t < T; ++t) {
        int ind_t = d * T + t;
        for (int r = 0; r < R; ++r) {
          lambda_model2_file << c << " " << r << " " << ind_t << " "
                             << result.lambda_missing(c, d, t, r) << " "
                             << lambda(c, d, t, r) << "\n";
        }
      }
    }
  }
  lambda_model2_file.close();
}

void run_models_missing_location(AppParams& app_params) {
  std::cout << "info_file = " << app_params.info_file << "\n";
  std::ifstream info_file(app_params.info_file, std::ios::in);
  ulong C, D, T, R, nb_regressors, nb_holidays_years;
  info_file >> T >> D >> R >> C >> nb_regressors >> nb_holidays_years;
  std::vector<int> daily_obs(D, 0);

  // printf("%ld %ld %ld %ld %ld %ld\ndaily_obs = ", T, D, R, C,
  // nb_regressors,
  //        nb_holidays_years);
  for (int d = 0; d < D; ++d) {
    info_file >> daily_obs[d];
    // printf("%d ", daily_obs[d]);
  }
  // printf("\n");
  info_file.close();
  printf("Read info_file\n");
  xt::xarray<int> nb_observations = xt::zeros<int>({C, D, T, R});
  ulong nb_obs = *max_element(daily_obs.begin(), daily_obs.end());
  xt::xarray<int> sample = xt::zeros<int>({C, D, T, R, nb_obs});
  xt::xarray<int> nb_arrivals = xt::zeros<int>({C, D, T, R});
  std::string aux_str;
  std::ifstream arrivals_file(app_params.arrivals_file, std::ios::in);
  do {
    getline(arrivals_file, aux_str);
    if (aux_str == "END") {
      break;
    }
    std::istringstream ss(aux_str);
    int t, d, r, c, j, val;
    ss >> t >> d >> r >> c >> j >> val;
    if (c > C || d > D || t > T || r > R || j >= nb_obs) {
      std::cout << "Error in arrivals file: c = " << c << ", d = " << d
                << ", t = " << t << ", r = " << r << ", j = " << j
                << ", val = " << val << "\n";
      std::cin.get();
    }
    sample(c, d, t, r, j) = val;
    nb_observations(c, d, t, r) += 1;
    nb_arrivals(c, d, t, r) += val;
  } while (true);
  arrivals_file.close();
  printf("Read arrivals_file\n");

  for (int c = 0; c < C; ++c) {
    for (int d = 0; d < D; ++d) {
      for (int t = 0; t < T; ++t) {
        for (int r = 0; r < R; ++r) {
          nb_observations(c, d, t, r) = daily_obs[d];
        }
      }
    }
  }

  auto type_region = std::vector<int>(R, -1);
  xt::xarray<double> regressors = xt::zeros<double>({nb_regressors, R});
  auto neighbors = std::vector<std::vector<int>>(R, std::vector<int>());
  xt::xarray<double> distance = xt::zeros<double>({R, R});

  std::ifstream neighbors_file(app_params.neighbors_file, std::ios::in);
  std::vector<double> pop(R, 0.0);
  while (true) {
    int ind, terrain_type, s, land_type;
    double lat, longi, dist;
    std::getline(neighbors_file, aux_str);
    if (aux_str == "END") {
      break;
    }
    // std::cout << aux_str << "\n";
    std::istringstream ss(aux_str);
    ss >> ind >> lat >> longi >> land_type;
    type_region[ind] = land_type;
    for (int j = 0; j < nb_regressors; ++j) {
      ss >> regressors(j, ind);
    }
    pop[ind] = regressors(0, ind);
    while (ss >> s >> dist) {
      distance(ind, s) = dist;
      neighbors[ind].push_back(s);
    }
  }
  neighbors_file.close();
  printf("Read neighbors_file\n");

  double sum_pop = std::accumulate(pop.begin(), pop.end(), 0.0);
  for (int r = 0; r < R; ++r) {
    pop[r] = pop[r] / sum_pop;
  }

  xt::xarray<int> sample_missing_arrivals = xt::zeros<int>({C, D, T, nb_obs});
  xt::xarray<int> nb_missing_arrivals = xt::zeros<int>({C, D, T});

  std::ifstream missing_file(app_params.missing_file, std::ios::in);
  do {
    getline(missing_file, aux_str);
    if (aux_str == "END") {
      break;
    }
    std::istringstream ss(aux_str);
    double d, r;
    int t, c, j, val;
    ss >> t >> d >> r >> c >> j >> val;

    sample_missing_arrivals(c, static_cast<int>(d), t, j) = val;
    nb_missing_arrivals(c, static_cast<int>(d), t) += val;
  } while (true);
  missing_file.close();
  printf("Read missing_arrivals file\n");
  std::cout << std::flush;

  xt::xarray<double> durations = 0.5 * xt::ones<double>({D, T});

  if (app_params.model == "all") {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = run_model1(nb_observations, nb_arrivals, nb_missing_arrivals,
                             durations);
    auto dt = std::chrono::high_resolution_clock::now() - t0;
    double run_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count() / 1e9;
    printf("Finished analytical model %f\n", run_time);
    std::cout << std::flush;
    run_regularized_model(app_params, nb_observations, nb_arrivals,
                          nb_missing_arrivals, sample, sample_missing_arrivals,
                          durations, neighbors);
    printf("Finished Regularized model\n");
    std::cout << std::flush;
    run_model_population(app_params, nb_observations, nb_arrivals,
                         nb_missing_arrivals, durations, sample,
                         sample_missing_arrivals, pop, result);
    printf("Finished population model\n");
    std::cout << std::flush;
  } else if (app_params.model == "analytical") {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = run_model1(nb_observations, nb_arrivals, nb_missing_arrivals,
                             durations);
    auto dt = std::chrono::high_resolution_clock::now() - t0;
    double run_time =
        std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count() / 1e9;
    printf("Finished analytical model %f\n", run_time);
    std::cout << std::flush;
  } else if (app_params.model == "regularized") {
    run_regularized_model(app_params, nb_observations, nb_arrivals,
                          nb_missing_arrivals, sample, sample_missing_arrivals,
                          durations, neighbors);
    printf("Finished Regularized model\n");
  } else if (app_params.model == "population") {
    auto result = run_model1(nb_observations, nb_arrivals, nb_missing_arrivals,
                             durations);
    run_model_population(app_params, nb_observations, nb_arrivals,
                         nb_missing_arrivals, durations, sample,
                         sample_missing_arrivals, pop, result);
    printf("Finished population model\n");
  }
}

xt::xarray<double> missing_time_analytical_p(
    AppParams& app_params, xt::xarray<int>& nb_observations,
    xt::xarray<int>& nb_arrivals, xt::xarray<int>& nb_missing_arrivals,
    xt::xarray<int>& sample, xt::xarray<int>& sample_missing_arrivals,
    xt::xarray<double>& durations) {
  ulong C = nb_observations.shape(0);
  ulong D = nb_observations.shape(1);
  ulong T = nb_observations.shape(2);
  ulong R = nb_observations.shape(3);
  xt::xarray<double> p_est = xt::zeros<double>({C, D, R});
  for (ulong c = 0; c < C; ++c) {
    for (ulong d = 0; d < D; ++d) {
      for (ulong r = 0; r < R; ++r) {
        double m0_cdr = nb_missing_arrivals(c, d, r);
        double m1_cdr = 0.0;
        for (ulong t = 0; t < T; ++t) {
          m1_cdr += nb_arrivals(c, d, t, r);
        }
        p_est(c, d, r) = m0_cdr / (m0_cdr + m1_cdr);
      }
    }
  }
  return p_est;
}

std::pair<xt::xarray<double>, xt::xarray<double>>
missing_time_analytical_lambdas(AppParams& app_params,
                                xt::xarray<int>& nb_observations,
                                xt::xarray<int>& nb_arrivals,
                                xt::xarray<int>& nb_missing_arrivals,
                                xt::xarray<int>& sample,
                                xt::xarray<int>& sample_missing_arrivals,
                                xt::xarray<double>& durations) {
  ulong C = nb_observations.shape(0);
  ulong D = nb_observations.shape(1);
  ulong T = nb_observations.shape(2);
  ulong R = nb_observations.shape(3);

  xt::xarray<double> lambda_est = xt::zeros<double>({C, D, T, R});
  xt::xarray<double> lambda_est_no_missing = xt::zeros<double>({C, D, T, R});
  for (ulong c = 0; c < C; ++c) {
    for (ulong d = 0; d < D; ++d) {
      for (ulong r = 0; r < R; ++r) {
        double m0_cdr = nb_missing_arrivals(c, d, r);
        double m1_cdr = 0.0;
        for (ulong t = 0; t < T; ++t) {
          m1_cdr += nb_arrivals(c, d, t, r);
        }
        for (ulong t = 0; t < T; ++t) {
          lambda_est_no_missing(c, d, t, r) =
              nb_arrivals(c, d, t, r) /
              (nb_observations(c, d, t, r) * durations(d, t));
          double m1_cdtr = nb_arrivals(c, d, t, r);
          double n_cdtr = nb_observations(c, d, t, r);
          lambda_est(c, d, t, r) =
              ((m0_cdr + m1_cdr) / m1_cdr) * (m1_cdtr / n_cdtr);
          lambda_est(c, d, t, r) /= durations(d, t);
          // printf(
          //     "m0_cdr = %f, m1_cdr = %f, m1_cdtr = %f, n_cdtr = %f, lambda
          //     =
          //     "
          //     "%f\n",
          //     m0_cdr, m1_cdr, m1_cdtr, n_cdtr, lambda_est(c, d, t, r));
          // std::cin.get();
        }
      }
    }
  }
  return std::make_pair(lambda_est, lambda_est_no_missing);
}

int idx(ulong i, ulong j, ulong ny) { return i * ny + j; }

void generate_artificial_missing_time_data(AppParams& app_params,
                                           ulong nb_obs) {
  ulong C = 1;
  ulong D = 7;
  ulong T = 48;
  ulong R = 100;
  ulong nx = 10;
  ulong ny = 10;

  std::vector<std::vector<ulong>> grid_neighbors(R, std::vector<ulong>());
  for (ulong i = 0; i < nx; ++i) {
    for (ulong j = 0; j < ny; ++j) {
      ulong r = idx(i, j, ny);
      // Up
      if (i > 0) {
        grid_neighbors[r].push_back(idx(i - 1, j, ny));
      }
      // Down
      if (i < nx - 1) {
        grid_neighbors[r].push_back(idx(i + 1, j, ny));
      }
      // Left
      if (j > 0) {
        grid_neighbors[r].push_back(idx(i, j - 1, ny));
      }
      // Right
      if (j < ny - 1) {
        grid_neighbors[r].push_back(idx(i, j + 1, ny));
      }
    }
  }
  ulong mid_x = nx / 2;
  ulong mid_y = ny / 2;
  std::vector<bool> is_blue(R, false);
  for (ulong i = 0; i < nx; ++i) {
    for (ulong j = 0; j < ny; ++j) {
      // is_blue[r] for lower-left and upper-right quadrants
      ulong r = idx(i, j, ny);
      if ((i < mid_x && j >= mid_y) || (i >= mid_x && j < mid_y)) {
        is_blue[r] = true;
      }
    }
  }

  std::vector<std::vector<std::pair<ulong, ulong>>> time_groups(
      4, std::vector<std::pair<ulong, ulong>>());
  std::vector<std::vector<size_t>> which_group(D, std::vector<size_t>(T, 4));
  for (ulong d = 0; d < D; ++d) {
    for (ulong t = 0; t < 12; ++t) {
      time_groups[0].push_back(std::make_pair(d, t));
      which_group[d][t] = 0;
    }
    for (ulong t = 12; t < 24; ++t) {
      time_groups[1].push_back(std::make_pair(d, t));
      which_group[d][t] = 1;
    }
    for (ulong t = 24; t < 36; ++t) {
      time_groups[2].push_back(std::make_pair(d, t));
      which_group[d][t] = 2;
    }
    for (ulong t = 36; t < 48; ++t) {
      time_groups[3].push_back(std::make_pair(d, t));
      which_group[d][t] = 3;
    }
  }

  xt::xarray<double> lambda_true = xt::zeros<double>({C, D, T, R});
  for (ulong c = 0; c < C; ++c) {
    for (ulong d = 0; d < D; ++d) {
      for (ulong t = 0; t < T; ++t) {
        for (ulong r = 0; r < R; ++r) {
          double base_lambda = 1.0;

          if (is_blue[r] && which_group[d][t] == 1) {
            base_lambda += 2.0;
          } else if (is_blue[r] && which_group[d][t] == 3) {
            base_lambda += 3.0;
          }
          lambda_true(c, d, t, r) = base_lambda;
        }
      }
    }
  }
  xt::xarray<double> prob_missing = xt::zeros<double>({C, D, R});
  for (ulong c = 0; c < C; ++c) {
    for (ulong d = 0; d < D; ++d) {
      for (ulong r = 0; r < R; ++r) {
        if (is_blue[r] && (d == 5 || d == 6)) {
          prob_missing(c, d, r) = 0.4;
        } else {
          prob_missing(c, d, r) = 0.1;
        }
      }
    }
  }
  xt::xarray<int> nb_observations = nb_obs * xt::ones<int>({C, D, T, R});
  xt::xarray<int> nb_arrivals = xt::zeros<int>({C, D, T, R});
  xt::xarray<int> nb_missing_arrivals = xt::zeros<int>({C, D, R});
  xt::xarray<int> sample = xt::zeros<int>({C, D, T, R, nb_obs});
  xt::xarray<int> sample_missing_arrivals = xt::zeros<int>({C, D, R, nb_obs});
  xt::xarray<double> durations = 0.5 * xt::ones<double>({D, T});

  std::default_random_engine generator;
  for (ulong c = 0; c < C; ++c) {
    for (ulong d = 0; d < D; ++d) {
      for (ulong t = 0; t < T; ++t) {
        for (ulong r = 0; r < R; ++r) {
          double lambda_ctr = lambda_true(c, d, t, r);
          double p_ctr = prob_missing(c, d, r);
          std::poisson_distribution<int> poisson_dist((1 - p_ctr) * lambda_ctr *
                                                      durations(d, t));
          for (ulong k = 0; k < nb_obs; ++k) {
            int val = poisson_dist(generator);
            sample(c, d, t, r, k) = val;
            nb_arrivals(c, d, t, r) += val;
          }
        }
      }
    }
  }
  for (ulong c = 0; c < C; ++c) {
    for (ulong d = 0; d < D; ++d) {
      for (ulong r = 0; r < R; ++r) {
        double p_ctr = prob_missing(c, d, r);
        double sum_lambda = 0.0;
        for (ulong t = 0; t < T; ++t) {
          sum_lambda += lambda_true(c, d, t, r) * durations(d, t);
        }
        std::poisson_distribution<int> poisson_dist(p_ctr * sum_lambda);
        for (ulong k = 0; k < nb_obs; ++k) {
          int val = poisson_dist(generator);
          sample_missing_arrivals(c, d, r, k) = val;
          nb_missing_arrivals(c, d, r) += val;
        }
      }
    }
  }

  std::pair<xt::xarray<double>, xt::xarray<double>> result_lambda =
      missing_time_analytical_lambdas(app_params, nb_observations, nb_arrivals,
                                      nb_missing_arrivals, sample,
                                      sample_missing_arrivals, durations);
  xt::xarray<double> lambda_est = result_lambda.first;
  xt::xarray<double> lambda_est_no_missing = result_lambda.second;

  xt::xarray<double> p_est = missing_time_analytical_p(
      app_params, nb_observations, nb_arrivals, nb_missing_arrivals, sample,
      sample_missing_arrivals, durations);

  double sum_diff = 0.0;
  for (ulong c = 0; c < C; ++c) {
    for (ulong d = 0; d < D; ++d) {
      for (ulong t = 0; t < T; ++t) {
        for (ulong r = 0; r < R; ++r) {
          sum_diff +=
              std::abs(lambda_est(c, d, t, r) - lambda_true(c, d, t, r)) /
              lambda_true(c, d, t, r);
        }
      }
    }
  }
  double avg_diff_lambda = sum_diff / (C * D * T * R);
  sum_diff = 0.0;
  for (ulong c = 0; c < C; ++c) {
    for (ulong d = 0; d < D; ++d) {
      for (ulong r = 0; r < R; ++r) {
        double p_true = prob_missing(c, d, r);
        sum_diff += std::abs(p_est(c, d, r) - p_true) / p_true;
      }
    }
  }
  double avg_diff_p = sum_diff / (C * D * R);
  std::stringstream out_missing_file;
  out_missing_file << "results/missing_time_lambda_nbobs_" << nb_obs << ".dat";
  std::ofstream arq_lambda(out_missing_file.str(), std::ios::out);
  for (ulong c = 0; c < C; ++c) {
    for (ulong d = 0; d < D; ++d) {
      for (ulong t = 0; t < T; ++t) {
        int index_t = d * T + t;
        for (ulong r = 0; r < R; ++r) {
          arq_lambda << c << " " << r << " " << index_t << " "
                     << lambda_est(c, d, t, r) << "\n";
        }
      }
    }
  }
  arq_lambda.close();
  std::stringstream out_no_missing_file;
  out_no_missing_file << "results/missing_time_lambda_no_missing_nbobs_"
                      << nb_obs << ".dat";
  std::ofstream arq_lambda_no_missing(out_no_missing_file.str(), std::ios::out);
  for (ulong c = 0; c < C; ++c) {
    for (ulong d = 0; d < D; ++d) {
      for (ulong t = 0; t < T; ++t) {
        int index_t = d * T + t;
        for (ulong r = 0; r < R; ++r) {
          arq_lambda_no_missing << c << " " << r << " " << index_t << " "
                                << lambda_est_no_missing(c, d, t, r) << "\n";
        }
      }
    }
  }
  arq_lambda_no_missing.close();
  printf("%ld\t%.3f\t%.3f\n", nb_obs, avg_diff_lambda, avg_diff_p);
}

void run_models_missing_time(AppParams& app_params) {
  std::vector<ulong> nb_obs_values{10, 100, 500, 1000, 2000};
  printf("nb_obs\tavg_diff_lambda\tavg_diff_p\n");
  for (ulong nb_obs : nb_obs_values) {
    generate_artificial_missing_time_data(app_params, nb_obs);
  }
  // ulong C, D, T, R, nb_obs;
  // std::ifstream theo_lambda("Artificial/theoretical_lambda.dat",
  // std::ios::in); theo_lambda >> C >> D >> T >> R >> nb_obs; printf("C = %ld
  // D = %ld T = %ld R = %ld nb_obs = %ld\n", C, D, T, R, nb_obs);
  // xt::xarray<double> theoretical_lambda = xt::zeros<double>({C, D, T, R});
  // while (!theo_lambda.eof()) {
  //   int c, d, t, r;
  //   double val;
  //   theo_lambda >> c >> d >> t >> r >> val;
  //   printf("c = %d d = %d t = %d r = %d val = %f\n", c, d, t, r, val);
  //   theoretical_lambda(c, d, t, r) = val;
  // }
  // theo_lambda.close();
  // printf("Finished reading theoretical lambda\n");
  // xt::xarray<int> nb_observations = nb_obs * xt::ones<int>({C, D, T, R});
  // xt::xarray<int> nb_arrivals = xt::zeros<int>({C, D, T, R});
  // xt::xarray<int> nb_missing_arrivals = xt::zeros<int>({C, D, R});
  // xt::xarray<int> sample = xt::zeros<int>({C, D, T, R, nb_obs});
  // xt::xarray<int> sample_missing_arrivals = xt::zeros<int>({C, D, R,
  // nb_obs});

  // std::ifstream sample_file("Artificial/calls.dat", std::ios::in);
  // std::string aux_str;
  // do {
  //   getline(sample_file, aux_str);
  //   if (aux_str == "END") {
  //     break;
  //   }
  //   std::istringstream ss(aux_str);
  //   int c, t, d, r, j, val;
  //   ss >> t >> d >> r >> c >> j >> val;
  //   printf("t = %d d = %d r = %d c = %d j = %d val = %d\n", t, d, r, c, j,
  //   val); std::cin.get(); sample(c, d, t, r, j) = val; nb_arrivals(c, d, t,
  //   r) += val;
  // } while (true);
  // sample_file.close();
  // printf("Finished reading sample data\n");
  // std::ifstream missing_file("Artificial/missing.dat", std::ios::in);
  // do {
  //   getline(missing_file, aux_str);
  //   if (aux_str == "END") {
  //     break;
  //   }
  //   std::istringstream ss(aux_str);
  //   int c, d, r, j, val;
  //   ss >> d >> r >> c >> j >> val;

  //   sample_missing_arrivals(c, d, r, j) = val;
  //   nb_missing_arrivals(c, d, r) += val;
  // } while (true);
  // printf("Finished reading missing data\n");
  // xt::xarray<double> durations = 0.5 * xt::ones<double>({D, T});
  // xt::xarray<double> lambda_est = missing_time_analytical_lambdas(
  //     app_params, nb_observations, nb_arrivals, nb_missing_arrivals,
  //     sample, sample_missing_arrivals, durations);
  // double sum_diff = 0.0;
  // for (ulong c = 0; c < C; ++c) {
  //   for (ulong d = 0; d < D; ++d) {
  //     for (ulong t = 0; t < T; ++t) {
  //       for (ulong r = 0; r < R; ++r) {
  //         sum_diff +=
  //             std::abs(lambda_est(c, d, t, r) - theoretical_lambda(c, d, t,
  //             r));
  //       }
  //     }
  //   }
  // }
  // double avg_diff = sum_diff / (C * D * T * R);
  // std::cout << "Average difference lambda = " << avg_diff << "\n";
  // xt::xarray<double> p_est = missing_time_analytical_p(
  //     app_params, nb_observations, nb_arrivals, nb_missing_arrivals,
  //     sample, sample_missing_arrivals, durations);
}

AppParams load_options(int argc, char* argv[]) {
  std::string config_file;
  po::variables_map vm;
  // Declare a group of options that will be
  // allowed only on command line
  po::options_description generic("Generic Options");
  generic.add_options()("help,h", "Display this help message.")(
      "file,f", po::value<std::string>()->default_value(""),
      "Path to configuration file.");

  po::options_description config("Configuration");
  using doubles = std::vector<double>;
  doubles test_weights;
  config.add_options()(
      "EPS,E", po::value<double>()->default_value(1e-5),
      "Epsilon for feasibility and convergence checks. Default = 1e-5")(
      "sigma,s", po::value<double>()->default_value(0.5),
      "Sigma parameter of armijo step. Default = 0.5")(
      "max_iter,I", po::value<int>()->default_value(30),
      "Max number of iterations used in stopping criterion. Default = 30")(
      "lower_lambda,L", po::value<double>()->default_value(1e-6),
      "Lower bound on decision variables for both models. Default = 1e-6")(
      "beta_bar,B", po::value<double>()->default_value(2.0),
      "Initial step size for projected gradient. Default = 2.0")(
      "mc_samples,S", po::value<ulong>()->default_value(30),
      "Number of Monte Carlo samples. Default = 30")(
      "test_weights", po::value<doubles>(&test_weights)->multitoken(),
      "List of weights for the regularized model.")(
      "model", po::value<std::string>()->default_value(""),
      "Type of model to execute: all | analytical | regularized | "
      "population")("info_file,i", po::value<std::string>()->default_value(""),
                    "Path to file with general information about the model.")(
      "arrivals_file,a", po::value<std::string>()->default_value(""),
      "Path to file with arrivals data. Default = ''")(
      "neighbors_file,n", po::value<std::string>()->default_value(""),
      "Path to file with neighbors data. Default = ''")(
      "missing_file,N", po::value<std::string>()->default_value(""),
      "Path to file with missing arrivals data. Default = ''")(
      "mn_samples_file,m", po::value<std::string>()->default_value(""),
      "Path to file with MN samples. Default = ''");

  po::options_description cmdline_options;
  cmdline_options.add(generic).add(config);

  po::options_description config_file_options;
  config_file_options.add(config);

  po::options_description visible("Allowed Options");
  visible.add(generic).add(config);

  store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
  notify(vm);

  config_file = vm["file"].as<std::string>();
  std::ifstream ifs(config_file);
  if (config_file != "" && ifs) {
    store(parse_config_file(ifs, config_file_options), vm);
    notify(vm);
  } else if (config_file != "" && !ifs) {
    printf("Could not open config file: %s\n", config_file.c_str());
  }
  if (vm.count("help")) {
    std::cout << visible << "\n";
    exit(0);
  }

  AppParams app_params;
  app_params.EPS = vm["EPS"].as<double>();
  app_params.sigma = vm["sigma"].as<double>();
  app_params.max_iter = vm["max_iter"].as<int>();
  app_params.lower_lambda = vm["lower_lambda"].as<double>();
  app_params.beta_bar = vm["beta_bar"].as<double>();
  app_params.S = vm["mc_samples"].as<ulong>();
  app_params.test_weights = test_weights;
  app_params.model = vm["model"].as<std::string>();
  app_params.info_file = vm["info_file"].as<std::string>();
  app_params.arrivals_file = vm["arrivals_file"].as<std::string>();
  app_params.neighbors_file = vm["neighbors_file"].as<std::string>();
  app_params.missing_file = vm["missing_file"].as<std::string>();
  app_params.mn_samples_file = vm["mn_samples_file"].as<std::string>();
  return app_params;
}

int main(int argc, char* argv[]) {
  AppParams app_params = load_options(argc, argv);

  // run_models_missing_time(app_params);
  // std::cin.get();

  run_models_missing_location(app_params);
  return 0;
}
