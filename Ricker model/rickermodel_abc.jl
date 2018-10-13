
doc"""
    set_up_abcmcmc_problem(use_sim_data::Bool=true;nbr_of_unknown_parameters::Int64=2,
  prior_dist::String = "Uniform", adaptiveon::Bool = true, ploton::Bool = false,
  nbr_of_cores::Int64 = 8, T::Int64 = 50,x0::Float64 = 7.,
  print_interval::Int64 = 500, alg::String="abcmcmc", nbr_summary_stats::Int64=5,
  eps::Float64=0.1)

set_up_abcmcmc_problem sets the parameters for the problem defined by the inputs.

# Output
* `problem`: An instance of the type ProblemABCMCMC cointaining all information for the problem
"""
function set_up_abcmcmc_problem(use_sim_data::Bool=true;nbr_of_unknown_parameters::Int64=2,
  prior_dist::String = "Uniform", adaptiveon::Bool = true, ploton::Bool = false,
  nbr_of_cores::Int64 = 8, T::Int64 = 200,x0::Float64 = 7.,
  print_interval::Int64 = 500, alg::String="abcmcmc", nbr_summary_stats::Int64=5,
  start::String = "random")


  # todo: set parameters and objects

  # set algorithm parameters
  theta_true = log([44.7; 10; 0.3]) # r φ σ
  theta_0 =  log([3;3;0.5])
  #prior_parameters = [0 5; 0 4;-10 2]
  prior_parameters = [2 5; 1.61 3;-3 -0.22]
  theta_known = [NaN]

  eps = [30*ones(10000);20*ones(10000);10*ones(10000); 5*ones(10000); 3*ones(10000); 2*ones(10000); 1.5*ones(500000)]

  # create instance of AlgorithmParametersABCMCMC (set parameters to default values)
  alg_param = AlgorithmParametersABCMCMC(length(eps),length(eps)-200000,alg,print_interval,nbr_summary_stats,eps,start, ones(nbr_summary_stats)) # hardcoded nbr of summary stats)

  # create instance of ModelParameters, all theta paramters are on log-scale
  model_param = ModelParameters(theta_true,theta_known,theta_0)

  # set data
  if use_sim_data
    y = generate_data_ricker_model(T, exp(theta_true[1]), exp(theta_true[2]), exp(theta_true[3]), x0, ploton)
  end

  # create instance of Data
  data = Data(y)


  # create instance of PriorDistribution
  prior = PriorDistribution(prior_dist, prior_parameters)

  # create instance of AdaptiveUpdate
  adaptive_update = AMUpdate(eye(length(theta_0)), 2.4/sqrt(length(theta_0)), 1., 0.7, 50)

  # return the an instance of Problem
  return ProblemABCMCMC(data, alg_param, model_param, adaptive_update, prior, "log", "uniform", "original")

end


function generate_data_ricker_model(T::Int64,r::Float64,phi::Float64,sigma::Float64,x0::Float64=7.0,ploton::Bool=false)

  e = rand(Normal(0,sigma),T)
  y = zeros(T)
  x = zeros(T)

  # first iteration
  x[1] = r*x0*exp(-x0+e[1])

  y[1] = rand(Poisson(phi*x[1]))

  @simd for t = 2:T
    @inbounds x[t] = r*x[t-1]*exp(-x[t-1]+e[t])
    y[t] = rand(Poisson(phi*x[t]))
  end

  return y
end



function sample_from_prior(prior_dist::PriorDistribution)

  theta_hat = zeros(size(prior_dist.prior_parameters,1))
  if prior_dist.dist == "Uniform"
    for i = 1:size(prior_dist.prior_parameters,1)
      theta_hat[i] = rand(Uniform(prior_dist.prior_parameters[i,1], prior_dist.prior_parameters[i,2]))
    end
  end

  return theta_hat

end


function  evaluate_prior(theta_star::Vector, prior_parameters::Matrix, dist_type::String = "Uniform")

  # set start value for loglik
  log_likelihood = 0.

  if dist_type == "Uniform"
    for i = 1:length(theta_star)
      # Update loglik, i.e. add the loglik for each model paramter in theta
      log_likelihood = log_likelihood + log_unifpdf( theta_star[i], prior_parameters[i,1], prior_parameters[i,2] )
    end
  else
    # add other priors
  end

  return log_likelihood # return log_lik

end

function generate_data(N::Int64, theta_star::Vector, theta_known::Vector)
  return generate_data_ricker_model(N,exp(theta_star[1]),exp(theta_star[2]),exp(theta_star[3]))
end

function calc_summary(y::Vector)

  s1 = autocor(y,[5],demean=false)[1]

  s2 = log(var(y))

  s3 = mean(y)

  s5 = length(find(x -> x == 0 ,y))

  s6 = maximum(y)

  return [s1;s2;s3;s5;s6]
#  return [s1;s2;s5;s6]

end
