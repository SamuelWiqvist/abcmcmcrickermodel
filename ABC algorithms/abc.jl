# This file contains funtions and help functions for the ABC-MCMC algorithm and the
# DA-ABC-MCMC algorithm.

# Load packages:

using Distributions
using DataFrames
using StatsBase
using PyPlot
using KernelDensity

include(pwd()*"/adaptive updating algorithms/adaptiveupdate.jl")

# Types:

"Type for prior distribution"
type PriorDistribution
  dist::String
  prior_parameters::Array{Float64}
end

"Type for the data"
type Data
  y::Array{Float64}
end

"Pararameters for the model"
type ModelParameters
  theta_true::Array{Float64} # [log(r) log(phi) log(sigma)]
  theta_known::Array{Float64} # NaN
  theta_0::Array{Float64} # [log(r_0) log(phi_0) log(sigma_0)]
end

"type for the algorithm parameters for the ABC-MCMC algorithm"
type AlgorithmParametersABCMCMC
  R::Int64
  burn_in::Int64
  alg::String
  print_interval::Int64
  nbr_summary_stats::Int64
  eps::Vector
  start::String
  w::Vector
end

"Type for the problem (including algorithm parameters) for the PMCMC algorithm"
type ProblemABCMCMC
  data::Data
  alg_param::AlgorithmParametersABCMCMC
  model_param::ModelParameters
  adaptive_update::AdaptationAlgorithm
  prior_dist::PriorDistribution
  parameter_transformation::String
  abc_kernel::String
  abcmcmctype::String
end

"Type for the results"
type Result
  Theta_est::Array{Float64}
  accept_vec::Array{Float64}
  prior_vec::Array{Float64}
end


# Algorithms:

doc"""
    abcmcmc(problem::ProblemABCMCMC, sample_from_prior, evaluate_prior, generate_data, calc_summary)

The ABC-MCMC algorithm.
"""
function abcmcmc(problem::ProblemABCMCMC, sample_from_prior, evaluate_prior, generate_data, calc_summary)

  # data
  y = problem.data.y
  y_star = copy(y)
  length_data = length(y) # length of data set

  # algorithm parameters
  R = problem.alg_param.R # number of iterations
  burn_in = problem.alg_param.burn_in # burn in
  print_interval = problem.alg_param.print_interval # print accaptance rate and covarince function ever print_interval:th iteration
  start = problem.alg_param.start

  # model parameters
  theta_true = problem.model_param.theta_true # [log(r) log(phi) log(sigma)]
  theta_known = problem.model_param.theta_known # NaN
  theta_0 = problem.model_param.theta_0 # [log(r_0) log(phi_0) log(sigma_0)]

  # abc kernel
  abc_kernel = problem.abc_kernel

  # transformation of parameter space
  parameter_transformation = problem.parameter_transformation

  # ABC MCMC type
  abcmcmctype  = problem.abcmcmctype

  # ABC parameters
  nbr_summary_stats = problem.alg_param.nbr_summary_stats
  s = zeros(nbr_summary_stats)
  s_star = zeros(nbr_summary_stats)
  eps = problem.alg_param.eps
  w =  problem.alg_param.w
  s_matrix = zeros(length(s_star), R)

  # pre-allocate matricies and vectors
  Theta = zeros(length(theta_0),R)
  loglik = zeros(R)
  accept_vec = zeros(R)
  prior_vec = zeros(R)
  theta_star = zeros(length(theta_0))
  a_log = 0.

  # training data
  X = zeros(length(theta_0),R)
  Y = zeros(R)

  # parameters for adaptive update
  adaptive_update_params = set_adaptive_alg_params(problem.adaptive_update, length(theta_0),Theta[:,1], R)

  # parameters for prior dist
  dist_type = problem.prior_dist.dist
  prior_parameters = problem.prior_dist.prior_parameters

  # print information at start of algorithm
  @printf "Starting ABC-MCMC estimating %d parameters\n" length(theta_true)
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist

  # first iteration
  @printf "Iteration: %d\n" 1
  @printf "Covariance:\n"
  print_covariance(problem.adaptive_update,adaptive_update_params, 1)

  s = calc_summary(y,y)

  # initial iteration
  if start == "random"
    while true
      theta_star = sample_from_prior(problem.prior_dist)
      y_star = generate_data(length_data, theta_star,theta_known)
      s_star = calc_summary(y_star,y)
      if ρ(s,s_star,w) < eps[1]
        break
      end
    end
  else
    theta_star = theta_0
  end

  dist_vec = zeros(R)
  ind_vec = zeros(R)
  Theta[:,1] = theta_star
  accept_vec[1] = 1

  @printf "Start value:\n"
  show(theta_star)

  X[:,1] = theta_star
  Y[1] = 1

  s_matrix[:,1] = calc_summary(y_star,y)

  for r = 2:R

    # set print_on to false, only print each print_interval:th iteration
    print_on = false

    # print acceptance rate for the last print_interval iterations
    if mod(r-1,print_interval) == 0
      print_on = true # print ESS and Nbr resample each print_interval:th iteration
      # print progress
      @printf "Percentage done: %.2f %% \n" 100*(r-1)/R
      # print accaptace rate
      @printf "Acceptance rate on iteration %d to %d is %.4f %% \n" r-print_interval r-1  (sum(accept_vec[r-print_interval:r-1])/( r-1 - (r-print_interval) ))*100
      # print covaraince function
      @printf "Covariance:\n"
      print_covariance(problem.adaptive_update,adaptive_update_params, r)
      # print threshold
      @printf "Threshold: %.4f \n" eps[r]
    end

    # Gaussian random walk
    (theta_star, ) = gaussian_random_walk(problem.adaptive_update, adaptive_update_params, Theta[:,r-1], r)

    # store theta_star
    X[:,r] = theta_star


    # compute accaptance probability

    # Generate data
    y_star = generate_data(length_data, theta_star, theta_known)

    # Compute summary stats
    s_star = calc_summary(y_star,y)

    abc_likelihood_star = abckernel(s,s_star,w,eps[r], abc_kernel)
    abc_likelihood_old = abckernel(s,s_matrix[:,r-1],w,eps[r], abc_kernel)

    jacobian_log_star = jacobian(theta_star, parameter_transformation)
    jacobian_log_old = jacobian(Theta[:,r-1], parameter_transformation)

    prior_log_star = evaluate_prior(theta_star,prior_parameters)
    prior_log_old = evaluate_prior(Theta[:,r-1],prior_parameters)

    if abcmcmctype == "general"
      a_log = log(abc_likelihood_star) + prior_log_star +  jacobian_log_star - (log(abc_likelihood_old) +  prior_log_old + jacobian_log_old)
    else abcmcmctype == "original" && abc_kernel == "uniform"
      a_log = log(abc_likelihood_star) + prior_log_star +  jacobian_log_star - (prior_log_old + jacobian_log_old)
    end

    #a_log = log(abc_likelihood_star) + prior_log_star  - (log(abc_likelihood_old) +  prior_log_old)

    u_log = log(rand())
    accept =  u_log < a_log
    Y[r] = abc_likelihood_star # store val for likelihood approx. from the ABC kernel

    # update chain
    if accept # the proposal is accapted
      Theta[:,r] = theta_star # update chain with new proposals
      accept_vec[r] = 1
      s_matrix[:,r] = s_star
    else
      s_matrix[:,r] = s_matrix[:,r-1]
      dist_vec[r] = dist_vec[r-1]
      ind_vec[r] = ind_vec[r-1]
      Theta[:,r] = Theta[:,r-1] # keep old values
    end

    # adaptation of covaraince matrix for the proposal distribution
    adaptation(problem.adaptive_update, adaptive_update_params, Theta, r,a_log)

  end

  @printf "Ending ABC-MCMC with adaptive RW estimating %d parameters\n" length(theta_true)
  @printf "Adaptation algorithm: %s\n" typeof(problem.adaptive_update)
  @printf "Prior distribution: %s\n" problem.prior_dist.dist

  cov_prop_kernel = get_covariance(problem.adaptive_update,adaptive_update_params, R)

  return return_results(Theta,accept_vec,prior_vec, problem,adaptive_update_params),X,Y, cov_prop_kernel, s_matrix

end

# Help functions:

doc"""
    log_unifpdf(x::Float64, a::Float64,b::Float64)

Computes log(unifpdf(x,a,b)).
"""
function log_unifpdf(x::Float64, a::Float64, b::Float64)

  if  x >= a && x<= b
    return -log(b-a);
  else
    return log(0);
  end

end

doc"""
    jacobian(theta::Vector, parameter_transformation::String)

Returnes log-Jacobian for transformation of parameter space.
"""
function jacobian(theta::Vector, parameter_transformation::String)

  if parameter_transformation == "none"
    return 0.
  elseif parameter_transformation == "log"
    return sum(theta)
  end

end



function abckernel(s::Vector, s_star::Vector, w::Vector, eps::Float64, abc_kernel::String)

  if abc_kernel == "uniform"
    return indacator(s , s_star, w, eps)
  elseif abc_kernel == "Gauss"
    error("The Gaussian kernel is not implemented")
  end

end


doc"""
    indacator(s::Vector, s_star::Vector, w::Vector, eps::Float64)

Indacator function for the ABC-MH updating step.
"""
function indacator(s::Vector, s_star::Vector, w::Vector, eps::Float64)

  if ρ(s,s_star,w) < eps
    return 1.
  else
    return 0.
  end

end

doc"""
    ρ(s::Vector, s_star::Vector, w::Vector)

Distance function.
"""
function ρ(s::Vector, s_star::Vector, w::Vector)

  Δs =  (s_star-s)
  dist = Δs'*inv(diagm(w.^2))*Δs
  return sqrt(dist[1])

end

doc"""
    ρ(s::Vector, s_star::Vector)

Distance function (w constant).
"""
function ρ(s::Vector, s_star::Vector)

  ρ(s, s_star, ones(length(s)))

end

doc"""
   predict(beta::Vector, theta::Vector)

Prediction for the probit regression model at theta.
"""
function predict(beta::Vector, theta::Vector)

  p_hat =  exp(theta'*beta)./(1+exp(theta'*beta))
  return p_hat[1]

end

doc"""
    return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params)

Constructs the return type of the resutls from the MCWM and PMCMC algorithm.
"""
function return_results(Theta,accept_vec,prior_vec)

    return Result(Theta, accept_vec, prior_vec)

end

doc"""
    return_results(Theta,loglik,accept_vec,prior_vec, problem,adaptive_update_params)

Constructs the return type of the resutls from the MCWH and PMCMC algorithm.
"""
function return_results(Theta,accept_vec,prior_vec, problem,adaptive_update_params)

  if typeof(problem.adaptive_update) == AMUpdate_gen
    return (Result(Theta, loglik, accept_vec, prior_vec), adaptive_update_params[6])
  else
    return Result(Theta, accept_vec, prior_vec)
  end

end

doc"""
    calc_mad(s::Matrix)
"""
function calc_mad(s::Matrix)

  nrow = size(s,1)
  w = zeros(nrow)

  for i = 1:nrow
    w[i] = mad(s[i,:])
  end

  return w

end


doc"""
    stratresample(p , N)

Stratified resampling.

Sample N times with repetitions from a distribution on [1:length(p)] with probabilities p. See [link](http://www.cornebise.com/julien/publis/isba2012-slides.pdf).
"""
function stratresample(p , N)

  p = p/sum(p)  # normalize, just in case...

  cdf_approx = cumsum(p)
  cdf_approx[end] = 1
  #I = zeros(N,1)
  indx = zeros(Int64, N)
  U = rand(N,1)
  U = U/N + (0:(N - 1))/N
  index = 1
  for k = 1:N
    while (U[k] > cdf_approx[index])
      index = index + 1
    end
    indx[k] = index
  end

  return indx

end
