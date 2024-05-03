functions {

  // Used in likelihood
  real lambert_arg(
    real tN,      // time probe data
    real y1,      // initial atp concentration
    real t1, 
    real ktime, 
    real keff) {

    // Theoretical result returns yN_hat + log(yN_hat)
    real result = (y1/keff)*exp((-(tN - t1)/ktime) + y1/keff);

    return result; 
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

    // If not including KP
    real keff = ( 1 + (( atp0 + adp0 ) * inv_KD))/inv_C1;
    real ktime = (1 / (gamma * m * inv_KT) ) * ( 1 + (( atp0 + adp0 ) * inv_KD));
    
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
  // array[M] real<lower=0> tau;           // time delay to microscope in s
  // real<lower=0.001> gamma_mu; 

  // real<lower = 0.0001> inv_C1_mu;                     
  // real<lower = 0.0001> inv_KD_mu;                     // KD in units of uM
  real<lower = 0> sigma;  

  array[M] real theta_gamma_array;
  array[M] real theta_inv_C1_array;
  array[M] real theta_inv_KD_array; 

  array[M] real theta_beta_array; 
  }

transformed parameters {
  array[M] real<lower = 1e-5> inv_KT_array; // if not including KP
  array[M] real<lower=0.001> gamma_array;
  array[M] real<lower=1e-10> inv_C1_array;
  array[M] real<lower=1e-5> inv_KD_array; 
  array[M] real<lower=100> beta_array; 

  for (i in 1:M){
    inv_C1_array[i] = 0.005 + 0.0005*theta_inv_C1_array[i];
    inv_KD_array[i] = 0.01 + 0.001*theta_inv_KD_array[i]; 
    gamma_array[i] = 0.1 + 0.01*theta_gamma_array[i]; 

    beta_array[i] = 1e4 + 1e3*theta_beta_array[i]; 

    inv_KT_array[i] = inv_C1_array[i] + inv_KD_array[i];
  }
}

model {
  // Priors for hyper parameters
  // inv_C1_mu ~ normal(0.005, 0.05);
  // inv_KD_mu ~ normal(0.01, 0.1);
  sigma ~ normal(1, 0.1);

  // gamma_mu ~ uniform(0.001, 0.5); 

  for (i in 1:M) {
    // Priors for parameters
    theta_inv_C1_array[M] ~ std_normal(); 
    theta_inv_KD_array[M] ~ std_normal(); 
    theta_gamma_array[M] ~ std_normal();
    theta_beta_array[M] ~ std_normal(); 

    real inv_C1 = inv_C1_array[M]; 
    real inv_KD = inv_KD_array[M]; 
    real inv_KT = inv_KT_array[M]; 
    real gamma = gamma_array[M]; 


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
    // real initial_atp = y1; 

    vector[2] params = parameter_calculation(initial_atp, adp0[i], p0[i], inv_C1, inv_KT, inv_KD, gamma); 
    real keff = params[1];
    real ktime = params[2];

    // ------------------ Calculate likelihood ------------------

    for (j in 1:length){
      for (k in 1:length){
        real tN = time_slice[j];
        real tK = time_slice[k];
        real yK = atp_slice[k];
        real lambert_argument =  lambert_arg(tN, yK, tK, ktime, keff);
        real theoretical_atp = keff*lambert_w0(lambert_argument); 
        atp_slice[j] ~ normal(theoretical_atp, sigma);

        // Exponential model
        real theoretical_atp_exp = y1*exp(-(tN - t1)/beta_array[i]); 
        atp_slice[j] ~ normal(theoretical_atp_exp, sigma);
      }
    }
  }
}

generated quantities {
  // Prior predictive check
  array[N] real lambert_argument_generated;
  array[N] real atp_generated;
  array[M] real keff_generated;
  array[M] real ktime_generated;
  array[M] real tau_generated;
  array[N] real atp_exp_generated;



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

    vector[2] params = parameter_calculation(initial_atp, adp0[i], p0[i], inv_C1_array[i], inv_KT_array[i], inv_KD_array[i], gamma_array[i]); 
    real keff = params[1];
    real ktime = params[2]; 

    // save generated menten coefficients
    keff_generated[i] = keff; 
    ktime_generated[i] = ktime; 
    tau_generated[i] = (ktime/keff)*(initial_atp - y1) + ktime*log(initial_atp/y1) - t1; 

    // ------------------ Generate posterior ------------------
    // print("new"); 
    for (j in 1:length){
      real tN = time_slice[j]; 
      if (i == 1){
        lambert_argument_generated[j] = lambert_arg(tN, y1, t1, ktime, keff);
        atp_generated[j] = normal_rng(keff*lambert_w0(lambert_argument_generated[j]), sigma);

        atp_exp_generated[j] = normal_rng(y1*exp(-(tN - t1)/beta_array[i]), sigma);

      } else {
        int index = end_indices[i-1] + j; 
        lambert_argument_generated[index] = lambert_arg(tN, y1, t1, ktime, keff);
        atp_generated[index] = normal_rng(keff*lambert_w0(lambert_argument_generated[index]), sigma);

        atp_exp_generated[index] = normal_rng(y1*exp(-(tN - t1)/beta_array[i]), sigma);
      }
    }
  }
}