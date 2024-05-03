data {
  int N;
  int M; 
  array[N] real<lower=0> atp0;
  array[N] real<lower=0> atp_t_0;
  array[N] real<lower=0> adp0;
  array[N] real<lower=0> p0;
  array[M] real<lower=0> t;
}

generated quantities {
  // Parameters
  // real<lower=0> KT;
  // real<lower=0> KD; 
  // real<lower=0> KP; 
  // real<lower=0> Ktime; 
  real<lower=0> tau;
  
  // KT = lognormal_rng(5, 1);
  // KD = lognormal_rng(5, 1);
  // KP = lognormal_rng(5, 1);
  // Ktime = lognormal_rng(5, 1);
  tau = normal_rng(10, 2)*60; //prior on time delay

  // Data
  array[N, M] real atp_ppc;
  array[N] real Ktime;  

  // put a gaussian on atp_t_0[i]; - measurement error

  // for each experiment 
  for (i in 1:N) {
    // tau[i] = - Ktime*log(atp_t_0[i]/atp0[i]); 
    Ktime[i] = - tau/log(atp_t_0[i]/atp0[i]); // get ktime based on tau 

    // time range
    for (j in 1:M){
        atp_ppc[i, j] = atp0[i]*exp(-(t[j] + tau)/Ktime[i]);
    }
  }
}