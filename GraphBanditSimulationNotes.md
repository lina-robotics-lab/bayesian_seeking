###  Graph Bandit Simulations

1. **Local TS Agent**

   Current issue: investigate how to set the $\hat{P}_s$ estimator, esp. how to set the exploration profile parameter $\beta_t$. The same thing will be useful in UCB. 

   Attempted Solution: Papers in the parallel bandit folder. Look for $\beta_t$ annotations.

   **Solution to UCB**: From [Intro to MAB] book, section 1.3.1, eq. (1.5)
   $$
   UCB_a(t)=\mu_a(t)+\sqrt{2\log(T)/n_t(a)}
   $$
   where $n_t(a)$ is the number of times arm $a$ being sampled.
   
   **Solution to TS**: see Bayesian_Normal.pdf. If we assume $X|\mu\sim\mathcal{N}(\mu,\sigma^2)$ for fixed $\sigma^2$, and $\mu\sim \mathcal{N}(\mu_0,\sigma_0^2)$ for fixed $\mu_0,\sigma_0^2$, then the Bayesian Update is
   $$
   \mu|X\sim\mathcal{N}(\mu_1,\sigma_1^2)\\
   \sigma_1^2 = \frac{1}{\sigma_0^{-2}+n\sigma^{-2}},~\mu_1 = \sigma_1^2(\mu_0\sigma_0^{-2}+\sum x_i\sigma^{-2})
   $$
   Especially, if we choose $\mu_0=0,\sigma_0^2=1$, then 
   $$
   \mu_1 = \frac{\sum x_i}{\sigma^2+n},~\sigma_1^2=\frac{1}{1+n\sigma^{-2}}
   $$
   Therefore, we can assume $\mu_s|X\sim\hat{Q}_{s}$ is specified by $X|\mu_s\sim\mathcal{N}(\mu_s,\sigma^2)$ for fixed $\sigma^2$, and $\mu_s\sim \mathcal{N}(\mu_0,\sigma_0^2)$ for fixed $\mu_0,\sigma_0^2$. Then when doing TS, we sample $\hat{\mu}_s$ from $\hat{Q}_s$, so that $\hat{\mu}_s$ represents an instantiation of $\mu_s$ and is later used in acquisition.
   
   I recommend a large $\sigma_0^2$ and small $\sigma^2$ so that $\sigma_1^2$ has a large initial value that encourages exploration in the early stages, while that decays quickly so that we do not over-explore. For example, if we expect $\mu_s\in[a,b]$, and want $\sigma_1^2$ to decay below $\epsilon$ after $m$ samples, then we can select $\mu_0=(a+b)/2,\sigma_0^2=(a-b)^2/4, \sigma^2 = \frac{m\epsilon}{1-\epsilon\sigma_0^2} $.