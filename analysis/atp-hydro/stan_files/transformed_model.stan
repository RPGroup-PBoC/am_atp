functions {

  // Approximation taken from https://www.sciencedirect.com/science/article/pii/S1369703X12000277#fig0005
  real lambert_approximation(
    real x      // lambert argument
    ) {
    // Theoretical result returns yN_hat + log(yN_hat)
    real result = 1.45869*log(1.2*x/(log(2.4*x/(log(1 + 2.4*x))))) - 0.45869*log(2*x/(log(1+2*x)));

    return result; 
  }

  // Used in likelihood
  real lambert_arg(
    real tN,      // time probe data
    real yM,      // initial atp concentration
    real tM, 
    real ktime, 
    real keff) {
    // Theoretical result returns yN_hat + log(yN_hat)
    real result = (yM/keff)*exp((-(tN - tM)/ktime) + yM/keff); 
 
    return result; 
  }

  // Used in likelihood
  vector non_dimensionalisation(
    real yN, 
    real tN,      // time probe data
    real yM,      // initial atp concentration
    real tM, 
    real ktime, 
    real keff) {

    // Theoretical result returns nondimensionalised lhs and rhs
    real transformed_y = (yN - yM)/keff + log(yN/yM); 
    real transformed_t = -(tN - tM)/ktime; 

    return [transformed_y, transformed_t]'; 
  }

  vector parameter_calculation(
    real atp0,      // initial atp concentration
    real adp0,      // initial adp concentration
    real p0,        // initial phosphate concentration 
    real inv_C1,
    real inv_KT,       
    real inv_KD,
    // real KP,
    real gamma) {

    // Theoretical result converting time to ATP
    real m = 1; // motor concentration explicitly coded here since it's constant. Can make this varied if data for multiple motor concentrations is collected.
    
    // // If including KP
    // real keff = C1 * ( 1 + ( atp0 + adp0 ) / KD + ( atp0 + p0 ) / KP );
    // real ktime = (KT / (gamma * m) ) * ( 1 + ( atp0 + adp0 ) / KD + ( atp0 + p0 ) / KP );

    // If not including KP
    real keff = ( 1 + (( atp0 + adp0 ) * inv_KD))/inv_C1;
    real ktime = ( 1 + (( atp0 + adp0 ) * inv_KD))/(gamma * m * inv_KT);
    
    return [keff, ktime]'; 
  }

}

data {
  int N; // number of timepoints in total, from flattened data
  array[N] real time; // time
  array[N] real atp; 

  int M; // total number of conditions

  array[M] int end_indices; 
  array[M] real atp0;
  array[M] real adp0;
  array[M] real p0;
}



parameters {
  real<lower=0.001> gamma; 
  real<lower = 10.0> KT;                     
  real<lower = 2.0> delta_K;                     // KD in units of uM
  real<lower = 0> sigma;  

  real<lower=0.001> gamma_exp; 
  real<lower = 10.0> KT_exp;                     
  real<lower = 2.0> delta_K_exp;                     // KD in units of uM
  real<lower = 0> sigma_exp;  
  }

transformed parameters {
  real<lower = 10.0> KD = KT + delta_K; 
  real<lower = 0.0> inv_KT = 1/KT;
  real<lower = 0.0> inv_KD = 1/KD; 
  real<lower = 0.0> inv_C1 = inv_KT - inv_KD; 

  real<lower = 10.0> KD_exp = KT_exp + delta_K_exp; 
  real<lower = 0.0> inv_KT_exp = 1/KT_exp;
  real<lower = 0.0> inv_KD_exp = 1/KD_exp; 
  real<lower = 0.0> inv_C1_exp = inv_KT_exp - inv_KD_exp; 
}

model {
  // Priors for parameters
  // Lambert Likelihood
  KT ~ normal(30, 10);
  delta_K ~ normal(30, 20);
  sigma ~ normal(5, 0.05);
  gamma ~ uniform(0.001, 0.5); 

  // Exponential Likelihood
  KT_exp ~ normal(30, 10);
  delta_K_exp ~ normal(50, 10);
  sigma_exp ~ normal(5, 0.05);
  gamma_exp ~ uniform(0.001, 0.5); 

  
  for (i in 1:M) {

    // ------------------ Declare variables and Slice ATP and time ------------------
    int length; // to store length of sliced atp and time data.

    if (i == 1){
        length = end_indices[1]; 
    } else {
        length = end_indices[i] - end_indices[i-1]; 
    }
    array[length] real atp_slice; 
    array[length] real time_slice; 

    if (i == 1){
        atp_slice = atp[:end_indices[1]];
        time_slice = time[:end_indices[1]];
    } else {
        atp_slice = atp[end_indices[i-1]+1:end_indices[i]];
        time_slice = time[end_indices[i-1]+1:end_indices[i]];
    }

    real min_atp = min(atp_slice); 
    real max_atp = max(atp_slice); 

    // ------------------ Assign to variables ------------------

    real initial_atp = atp0[i]; 

    vector[2] params = parameter_calculation(initial_atp, adp0[i], p0[i], inv_C1, inv_KT, inv_KD, gamma); 
    vector[2] params_exp = parameter_calculation(initial_atp, adp0[i], p0[i], inv_C1_exp, inv_KT_exp, inv_KD_exp, gamma_exp); 

    real keff = params[1];
    real ktime = params[2];

    real ktime_exp = params_exp[2];

    // ------------------ Calculate likelihood ------------------

    for (k in 1:length){
      real yK = atp_slice[k]; 
      real tK = time_slice[k]; 

      for (j in 1:length){
        real yN = atp_slice[j]; 
        real tN = time_slice[j]; 

        // ------------------ Using Lambert Function ------------------
        real lambert_argument =  lambert_arg(tN, yK, tK, ktime, keff);

        real theoretical_atp = keff*lambert_w0(lambert_argument); 
        atp_slice[j] ~ normal(theoretical_atp, sigma);

        // ------------------ Using Exponential ------------------
        atp_slice[j] ~ normal(yK*exp(-(tN - tK)/ktime_exp), sigma_exp);

        // ------------------ Using Lambert Approximation ------------------
        // real lambert_argument; 
        // lambert_argument =  lambert_arg(tN, yK, tK, ktime, keff);
        // real theoretical_atp = keff*lambert_approximation(lambert_argument); 
        // atp_slice[j] ~ normal(theoretical_atp, sigma);

        // ------------------ Using non dimensionalised lhs and rhs ------------------
        // vector[2] non_dimensionalised_vars = non_dimensionalisation(yN, tN, yK, tK, ktime, keff);
        // real non_dimensionalised_y = non_dimensionalised_vars[1]; 
        // real non_dimensionalised_t = non_dimensionalised_vars[2]; 
        // non_dimensionalised_y ~ normal(non_dimensionalised_t, sigma); 
      }
      break;
    }
  }
}

generated quantities {
  // Prior predictive check
  array[N] real lambert_argument_generated;
  array[N] real atp_generated;
  array[N] real atp_exp_generated;
  array[M] real keff_generated;
  array[M] real ktime_generated;
  array[M] real tau_generated;
  array[N] real nd_atp_generated;
  array[N] real nd_atp_measured; 
  array[N] real nd_time_generated;
  array[N] real square_error;


  for (i in 1:M) {

    // ------------------ Declare variables and Slice ATP and time ------------------
    int length; // to store length of sliced atp and time data.

    if (i == 1){
        length = end_indices[1]; 
    } else {
        length = end_indices[i] - end_indices[i-1]; 
    }
    array[length] real atp_slice; 
    array[length] real time_slice; 

    if (i == 1){
        atp_slice = atp[:end_indices[1]];
        time_slice = time[:end_indices[1]];
    } else {
        atp_slice = atp[end_indices[i-1]+1:end_indices[i]];
        time_slice = time[end_indices[i-1]+1:end_indices[i]];
    }

    // ------------------ Assign to variables ------------------
    real y1 = atp_slice[1]; 
    real t1 = time_slice[1]; 
    real initial_atp = atp0[i]; 

    vector[2] params = parameter_calculation(initial_atp, adp0[i], p0[i], inv_C1, inv_KT, inv_KD, gamma); 
    real keff = params[1];
    real ktime = params[2]; 

    // save generated menten coefficients
    keff_generated[i] = keff; 
    ktime_generated[i] = ktime; 
    tau_generated[i] = (ktime/keff)*(initial_atp - y1) + ktime*log(initial_atp/y1) - t1;

    vector[2] params_exp = parameter_calculation(initial_atp, adp0[i], p0[i], inv_C1_exp, inv_KT_exp, inv_KD_exp, gamma_exp); 
    real ktime_exp = params_exp[2]; 

    // ------------------ Generate posterior ------------------
    // print("new"); 
    real yK = atp_slice[1]; 
    real tK = time_slice[1]; 

    for (j in 1:length){
      real yN = atp_slice[j]; 
      real tN = time_slice[j];

      vector[2] non_dimensionalised_vars = non_dimensionalisation(yN, tN, yK, tK, ktime, keff);
      real non_dimensionalised_y = non_dimensionalised_vars[1]; 
      real non_dimensionalised_t = non_dimensionalised_vars[2]; 

      non_dimensionalised_y = normal_rng(non_dimensionalised_t, sigma); 

      real arg = lambert_arg(tN, y1, t1, ktime, keff);

      if (i == 1){
        // Using lambert function
        lambert_argument_generated[j] = arg;
        atp_generated[j] = normal_rng(keff*lambert_approximation(lambert_argument_generated[j]), sigma);
        
        square_error[j] = square(atp_generated[j] - yN); 

        nd_atp_generated[j] = non_dimensionalised_y; 
        nd_time_generated[j] = non_dimensionalised_t; 

        // Using exponential
        atp_exp_generated[j] = normal_rng(yK*exp(-(tN - tK)/ktime_exp), sigma_exp);

      } else {
        // Using lambert function
        int index = end_indices[i-1] + j; 
        lambert_argument_generated[index] = arg;
        atp_generated[index] = normal_rng(keff*lambert_approximation(lambert_argument_generated[index]), sigma);

        square_error[index] = square(atp_generated[index] - yN); 

        nd_atp_generated[index] = non_dimensionalised_y; 
        nd_time_generated[index] = non_dimensionalised_t; 

        // Using exponential
        atp_exp_generated[index] = normal_rng(yK*exp(-(tN - tK)/ktime_exp), sigma_exp);

        
      }

      
    }
  }
}