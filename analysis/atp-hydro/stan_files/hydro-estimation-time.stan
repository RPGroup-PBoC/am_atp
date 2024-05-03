functions {
  real time_theor(
    real atp,       // atp probe data
    real time,      // time probe data
    real atp0,      // initial atp concentration
    real adp0,      // initial adp concentration
    real p0,        // initial phosphate concentration 
    real KT,       
    real KD,
    real KP,
    real gamma) {

    // Theoretical result converting time to ATP
    real m = 1; // motor concentration explicitly coded here since it's constant. Can make this varied if data for multiple motor concentrations is collected.
    real keff = KT * ( 1 + ( atp0 + adp0 ) / KD + ( atp0 + p0 ) / KP ) / ( (1/KT) - (1/KD) - (1/KP) );
    real ktime = ((1/KT) - (1/KD) - (1/KP)) * keff / ( gamma * m );
    real result = ktime * ( ( ( atp0 - atp ) / keff ) +  log(atp0/atp) );

    return result <= 0 ? 0.00001 : result;
  }
}

data {
  int N;                     // Total number of data points over entire data set (number of curves * number of data points per curve)
  // array[N] int condition;
  array[N] real atp; 
  array[N] real time; 
  array[N] real atp0;
  array[N] real adp0;
  array[N] real p0;
  // array[N] real speed;
}



parameters {
  // real<lower=0> kcat;        // units of nm/s
  real log_KT;                  // KT in units of uM
  real log_KD;                  // KD in units of uM
  real log_KP;                  // KP in units of uM
  real<lower=0> log_gamma;          // units of 1/s
  real<lower=0> sigma_t;        // units of s
  real<lower=0> sigma;       // units of s 

  vector[N] ln_time;    // t in units of s. This is the theoretical time calculated from the ATP datapoint. This becomes the mean for the likelihood, hence coded in the parameter list. 
}


transformed parameters {
  real gamma = 10^log_gamma;
  real KT = 10^log_KT;
  real KD = 10^log_KD;
  real KP = 10^log_KP;

  vector[N] t = exp(ln_time);
}


model {
  // Priors for parameters - using uniform as uninformative priors
  // log_gamma ~ normal(0.0, 1.0);
  log_gamma ~ normal(0.0, 2.0);
  log_KT ~ normal(0.0, 50.0);
  log_KD ~ normal(0.0, 50.0);
  log_KP ~ normal(0.0, 500.0);
  sigma_t ~ normal(0.0, 5.0);

  // Typical time for each condition
  for (i in 1:N) {
    ln_time[i] ~ normal(log(time_theor(atp[i], time[i], atp0[i], adp0[i], p0[i], KT, KD, KP, gamma)), sigma_t);
  }

  // Likelihood 
  for (j in 1:N) {
    time[j] ~ normal(t[j], sigma);
  }
}


generated quantities {
  array[N] real time_model;

  for (i in 1:N) {
    time_model[i] = time_theor(atp[i], time[i], atp0[i], adp0[i], p0[i], KT, KD, KP, gamma);
  }
}