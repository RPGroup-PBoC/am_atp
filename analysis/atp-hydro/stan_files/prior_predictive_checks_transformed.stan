functions {
  // // Used in likelihood
  // real log_y_by_y0(
  //   real atp,       // atp probe data
  //   real time,      // time probe data
  //   real atp0,      // initial atp concentration
  //   real adp0,      // initial adp concentration
  //   real p0,        // initial phosphate concentration 
  //   real KT,       
  //   real KD,
  //   real KP,
  //   real gamma, 
  //   real tau) {

  //   // Theoretical result converting time to ATP
  //   real m = 1; // motor concentration explicitly coded here since it's constant. Can make this varied if data for multiple motor concentrations is collected.
  //   real keff = KT * ( 1 + ( atp0 + adp0 ) / KD + ( atp0 + p0 ) / KP ) / ( (1/KT) - (1/KD) - (1/KP) );
  //   real ktime = ((1/KT) - (1/KD) - (1/KP)) * keff / ( gamma * m );

  //   real result = (time + tau)/ktime - (atp - atp0)/keff;

  //   return result <= 0 ? 0.00001 : result;
  // }

  // Used in likelihood
  real log_y_by_y0(
    real time,      // time probe data
    real ytau,      // initial atp concentration
    real ktime, 
    real keff,
    real tau) {

    // Theoretical result converting time to log(y/ytau) + y/keff 
    real result = (tau - time)/ktime + ytau/keff;

    return result; 
  }

  vector parameter_calculation(
    real ytau, 
    real atp0,      // initial atp concentration
    real adp0,      // initial adp concentration
    real p0,        // initial phosphate concentration 
    real C1,       
    real KD,
    real KP,
    real tau) {

    // Theoretical result converting time to ATP
    real m = 1; // motor concentration explicitly coded here since it's constant. Can make this varied if data for multiple motor concentrations is collected.
    real keff = C1 * ( 1 + ( atp0 + adp0 ) / KD + ( atp0 + p0 ) / KP );
    real ktime = tau/(log(atp0/ytau) + (atp0 - ytau)/keff); 
    
    // real C2 = (ktime * m) / ( 1 + ( atp0 + adp0 ) / KD + ( atp0 + p0 ) / KP ); 

    // real result = (time + tau)/ktime - (atp - atp0)/keff;

    return [keff, ktime]'; 
  }

}

data {
  // int N; // total number of data points
  // array[N] real atp; 
  // array[N] real log_atp_nd;
  // array[N] real time; 
  // array[N] real atp0;
  // array[N] real adp0;
  // array[N] real p0;

  int N; // number of timepoints for each condition
  array[N] real time; // time

  int M; // total number of conditions
  // array[M] int indices; // starting point for each condition
  array[M] real ytau;
  array[M] real atp0;
  array[M] real adp0;
  array[M] real p0;
}



// parameters {
//   real log_KT;                  // KT in units of uM
//   real log_KD;                  // KD in units of uM
//   real log_KP;                  // KP in units of uM
//   real<lower=0> log_gamma;      // units of 1/s
//   real<lower=0> log_tau;        // units of s
//   real<lower=0> sigma_t;        // units of s
//   // real<lower=0> sigma;          // units of s 

//   // vector[N] log_z;               // A.U.
//   }

// transformed parameters {
//   real tau = 10^log_tau; 
//   real gamma = 10^log_gamma;
//   real KT = 10^log_KT;
//   real KD = 10^log_KD;
//   real KP = 10^log_KP;

//   // vector[N] z = exp(log_z);
// }


// model {
//   // Priors for parameters - using uniform as uninformative priors
//   log_tau ~ normal(0.0, 50.0);
//   log_gamma ~ normal(0.0, 2.0);
//   log_KT ~ normal(0.0, 25.0);
//   log_KD ~ normal(0.0, 25.0);
//   log_KP ~ normal(0.0, 500.0);
//   sigma_t ~ normal(0.0, 0.5);

//   // Likelihood 
//   for (i in 1:N) {
//     log_atp_nd[i] ~ normal(log_y_by_y0(atp[i], time[i], atp0[i], adp0[i], p0[i], KT, KD, KP, gamma, tau), sigma_t);
//   }
// }


generated quantities {
  // Priors 
  array[M] real tau; 
  real C1 = normal_rng(10000, 1); 
  real KD = normal_rng(100, 10); 
  real KP = normal_rng(1000, 10); 

  // declare menten constants; 
  real keff; 
  real ktime; 

  array[N, M] real log_y_ppc;

  for (i in 1:M) {
    tau[i] = normal_rng(5, 2)*60; // time delay between 0 and 10 minutes
    vector[2] params = parameter_calculation(ytau[i], atp0[i], adp0[i], p0[i], C1, KD, KP, tau[i]); 
    keff = params[1];
    ktime = params[2];

    print("keff: ", keff); 
    print("slope: ", 1/ktime); 

    for (j in 1:N){
      log_y_ppc[j, i] = log_y_by_y0(time[j], ytau[i], ktime, keff, tau[i]);
    }
  }
}