# run script for the ABC-MCMC 


include(pwd()*"/ABC algorithms/abc.jl")
include(pwd()*"/Ricker model/rickermodel_abc.jl")
include(pwd()*"/Ricker model/plotting.jl")

################################################################################
##            pre-runs
################################################################################

# set up
problem = set_up_abcmcmc_problem(ploton = true,
  start = "non-random", print_interval=10000)

PyPlot.figure()
PyPlot.plot(problem.data.y)

#problem.model_param.theta_0 = problem.model_param.theta_true  # start at true values

problem.model_param.theta_0 = mean(problem.prior_dist.prior_parameters,2)

#problem.data.y = Array(readtable("y_data_set_2.csv"))[:,1]
problem.data.y = Array(readtable("Ricker model/y_abc_ricker_200_obs.csv"))[:,1] #Array(readtable("y.csv"))[:,1]

## plot data

# plot data
text_size = 20
tick_size = 15

y_plot_2 = L"$y$"
y_plot_1 = L"time"

PyPlot.figure()
PyPlot.plot(problem.data.y)
PyPlot.xlabel(y_plot_1,fontsize=text_size)
PyPlot.ylabel(y_plot_2,fontsize=text_size)
PyPlot.xlabel(L"x",fontsize=text_size)


### new summary stats following woods


using MultivariateStats
using StatsBase


function calc_summary_woods(y_sim::Vector, y_obs::Vector)

  # statistics from data
  s1 = autocov(y_sim)[1:6]
  s2 = mean(y_sim)
  s3 = length(find(x -> x == 0 ,y_sim))

  # regression diff(y) on observations
  y_regression = diff(y_sim)
  y_regression = sort(y_regression)
  x_regression = diff(y_obs)
  x_regression = sort(x_regression)
  X_regression = [x_regression x_regression.^2 x_regression.^3] # construct design matrix

  s_regression = llsq(X_regression, y_regression, trans=false, bias=false)


  # autoregression
  y_auto = y_sim.^0.3 - mean(y_sim.^0.3)
  y_auto_regression = y_auto[2:end]
  X_auto_regression = [y_auto[1:end-1] y_auto[1:end-1].^2]

  s_auto_regression = llsq(X_auto_regression, y_auto_regression, trans=false, bias=false)


  return [s1;s2;s3;s_regression;s_auto_regression]


end

###                     pre-run


problem.alg_param.w = ones(13)
problem.alg_param.nbr_summary_stats = 13
problem.adaptive_update = AMUpdate(eye(3),1/sqrt(3), 1., 0.7, 25)

problem.alg_param.eps = [2000*ones(100000);]

problem.alg_param.eps = [1500*ones(100000);
                        1000*ones(100000);
                        800*ones(100000);
                        700*ones(100000);
                        600*ones(100000);
                        500*ones(100000);
                        400*ones(200000)]

problem.alg_param.R = length(problem.alg_param.eps)
problem.alg_param.burn_in = problem.alg_param.R-200000 #problem.alg_param.R - 200000

Random.seed!(1234)
res, X, Y, cov_prop_kernel, s_matrix =  abcmcmc(problem, sample_from_prior, evaluate_prior, generate_data, calc_summary_woods)

Theta = res.Theta_est
accept_vec = res.accept_vec
prior_vec = res.prior_vec
burn_in = problem.alg_param.burn_in
prior_parameters = problem.prior_dist.prior_parameters
theta_true = problem.model_param.theta_true

#analyse_results(Theta, accept_vec, prior_vec,theta_true, burn_in, prior_parameters)

# plot posterior for pre-run
PyPlot.figure()
PyPlot.plot(res.Theta_est[1,:])
PyPlot.plot(ones(size(res.Theta_est,2),1)*problem.model_param.theta_true[1], "k");
PyPlot.ylabel(L"log $r$")
PyPlot.figure()
PyPlot.plot(res.Theta_est[2,:])
PyPlot.plot(ones(size(res.Theta_est,2),1)*problem.model_param.theta_true[2], "k");
PyPlot.figure()
PyPlot.plot(res.Theta_est[3,:])
PyPlot.plot(ones(size(res.Theta_est,2),1)*problem.model_param.theta_true[3], "k");


PyPlot.figure()
PyPlot.subplot(311)
PyPlot.plot(res.Theta_est[1,:])
PyPlot.plot(ones(size(res.Theta_est,2),1)*problem.model_param.theta_true[1], "k");
PyPlot.ylabel(L"log $r$")
PyPlot.subplot(312)
PyPlot.plot(res.Theta_est[2,:])
PyPlot.plot(ones(size(res.Theta_est,2),1)*problem.model_param.theta_true[2], "k");
PyPlot.ylabel(L"log $\phi$")
PyPlot.subplot(313)
PyPlot.plot(res.Theta_est[3,:])
PyPlot.plot(ones(size(res.Theta_est,2),1)*problem.model_param.theta_true[3], "k");
PyPlot.ylabel(L"log $\sigma$")




# compute weigths
idx = find(x-> x == 1, res.accept_vec)
filter!(x-> x > problem.alg_param.burn_in, idx)

s_matrix = s_matrix[:, idx]

#corrplot(s_matrix)

#w = mean(s_matrix,2)
w = calc_mad(s_matrix)

#w = sqrt(diag(cov(s_matrix,2)))
################################################################################
##      run ABC-MCMC
################################################################################

problem.alg_param.w = w
problem.alg_param.nbr_summary_stats = 13
problem.adaptive_update = AMUpdate(eye(3),1/sqrt(3), 1., 0.7, 25)

#=
# @eps = 1.5
problem.alg_param.eps = [2000*ones(100000);
                        500*ones(100000);
                        100*ones(100000);
                        50*ones(100000);
                        25*ones(100000);
                        10*ones(100000);
                        5*ones(100000);
                        4*ones(400000)]


=#

# @eps = 1.5
problem.alg_param.eps = [2000*ones(10000);
                        500*ones(10000);
                        100*ones(10000);
                        50*ones(10000);
                        25*ones(10000);
                        10*ones(10000);
                        5*ones(10000);
                        4*ones(10000);
                        3*ones(10000);
                        2*ones(10000);
                        1.5*ones(400000)]

problem.alg_param.R = length(problem.alg_param.eps)
problem.alg_param.burn_in = problem.alg_param.R - 400000

Random.seed!(1234)
res, X, Y, cov_prop_kernel, s_matrix = @time abcmcmc(problem, sample_from_prior, evaluate_prior, generate_data, calc_summary_woods)

# plot results
Theta = res.Theta_est
accept_vec = res.accept_vec
prior_vec = res.prior_vec
burn_in = problem.alg_param.burn_in
prior_parameters = problem.prior_dist.prior_parameters
theta_true = problem.model_param.theta_true

analyse_results(Theta, accept_vec, prior_vec,theta_true, burn_in, prior_parameters)
