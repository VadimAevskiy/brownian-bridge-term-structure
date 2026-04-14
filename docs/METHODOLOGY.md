# Mathematical Methodology

> *Self-contained derivation of the discrete-time Brownian bridge term structure model.  
> Equation numbers follow Aevskiy & Chetverikov (2016), Applied Economics, 48(25), 2333--2340.*

---

## 1. Setting and Notation

A country plans to enter a monetary union at a known future date $T$. Define:

| Symbol | Meaning |
|:------:|---------|
| $R_t$ | Short-term interest rate in the monetary union |
| $r_t$ | Short-term interest rate in the domestic country |
| $z_t = r_t - R_t$ | Short-term interest rate spread |
| $T$ | Date of monetary union entry |
| $\sigma$ | Conditional volatility of the spread process |
| $\lambda$ | Market price of spread risk |
| $\varepsilon_t$ | i.i.d. standard normal shocks |

**No-arbitrage requirement.** At time $T$, both currencies merge. If $z_T \neq 0$, one could borrow at the lower rate, lend at the higher rate, and earn a riskless profit once the exchange rate is fixed. Hence any consistent model must ensure $z_T = 0$ a.s.

---

## 2. The Discrete-Time Brownian Bridge

To guarantee $z_T = 0$, we specify (Eq. 1):

$$z_{t+1} = \left(1 - \frac{1}{T - t}\right) z_t + \sigma\,\varepsilon_{t+1}$$

**Proof that $z_T = 0$.** Set $t = T - 1$:

$$z_T = \left(1 - \frac{1}{T - T + 1}\right) z_{T-1} + \sigma\,\varepsilon_T = (1 - 1)\,z_{T-1} + 0 = 0$$

where $\varepsilon_T = 0$ is imposed as a boundary condition. We further set $z_t = 0$ and $\varepsilon_t = 0$ for all $t > T$.

**Continuous-time analogue.** The process is the discrete version of the Brownian bridge SDE:

$$dx_t = -\frac{x_t}{T - t}\,dt + \sigma\,dw_t$$

which satisfies $\lim_{t \to T} x_t = 0$ a.s. (Karatzas and Shreve 1991, Ch. 5).

---

## 3. Pricing Kernel

Following Backus, Foresi, and Telmer (2000), the domestic bond pricing kernel decomposes as:

$$\log m_{d,t} = \log m_{e,t} + \log m_t \qquad \text{(Eq. 2)}$$

where $m_{e,t}$ is the euro pricing kernel and $m_t = m(z_{t-1},\,\varepsilon_t)$ captures domestic spread risk. We specify (Eq. 5):

$$-\log m_{t+1} = \begin{cases} \lambda^2/2 + z_t + \lambda\,\varepsilon_{t+1} & t = 0, 1, \ldots, T-2 \\ z_t & t = T - 1 \\ 0 & t \geq T \end{cases}$$

This ensures spread risk vanishes at and beyond $T - 1$.

---

## 4. Domestic Discount Factor and Affine Conjecture

The $(n+1)$-period domestic discount factor satisfies the recursion (Eq. 6):

$$d_t^{n+1} = E_t\!\left[m_{t+1} \cdot d_{t+1}^n\right]$$

with $d_t^0 = 1$. We conjecture the affine form (Eq. 7):

$$-\log d_t^n = A_{n,t} + B_{n,t}\,z_t$$

**Initial conditions.** From $d_t^0 = 1$: $A_{0,t} = B_{0,t} = 0$. From the log-normality of $m_{t+1}$ and $d_t^0 = 1$: $A_{1,t} = 0$ and $B_{1,t} = 1$.

---

## 5. Derivation of the Double Recursion

For $0 \leq t \leq T - 2$, we expand the log of the pricing equation's integrand:

$$\log\bigl(m_{t+1} \cdot d_{t+1}^n\bigr) = -\frac{\lambda^2}{2} - z_t - \lambda\,\varepsilon_{t+1} - A_{n,t+1} - B_{n,t+1}\,z_{t+1}$$

Substituting the Brownian bridge for $z_{t+1}$:

$$= -\Bigl[\frac{\lambda^2}{2} + A_{n,t+1}\Bigr] - \Bigl[1 + B_{n,t+1}\!\left(1 - \frac{1}{T-t}\right)\Bigr] z_t - \bigl[\lambda + B_{n,t+1}\,\sigma\bigr]\,\varepsilon_{t+1}$$

This is Gaussian in $\varepsilon_{t+1}$ with:

$$\text{Mean} = -\Bigl[\frac{\lambda^2}{2} + A_{n,t+1}\Bigr] - \Bigl[1 + B_{n,t+1}\!\left(1 - \frac{1}{T-t}\right)\Bigr] z_t$$

$$\text{Variance} = \bigl(\lambda + B_{n,t+1}\,\sigma\bigr)^2$$

Applying the log-normal expectation formula $E\!\left[e^X\right] = e^{\mu + \sigma^2/2}$:

$$-\log d_t^{n+1} = \underbrace{\Bigl[A_{n,t+1} + \frac{\lambda^2}{2} - \frac{(\lambda + B_{n,t+1}\,\sigma)^2}{2}\Bigr]}_{A_{n+1,t}} + \underbrace{\Bigl[1 + B_{n,t+1}\!\left(1 - \frac{1}{T-t}\right)\Bigr]}_{B_{n+1,t}} z_t$$

This confirms the affine conjecture and yields the recursive formulas:

$$\boxed{B_{n+1,t} = 1 + B_{n,t+1}\!\left(1 - \frac{1}{T-t}\right)} \qquad \text{(Eq. 9)}$$

$$\boxed{A_{n+1,t} = A_{n,t+1} + \frac{\lambda^2}{2} - \frac{(\lambda + B_{n,t+1}\,\sigma)^2}{2}} \qquad \text{(Eq. 8)}$$

valid for $n + 1 \leq T - t$ and $t \leq T - 1$.

---

## 6. Closed-Form Solution for B

### Case 1: $n \leq T - t$

Rewrite the recursion as (Eq. 15):

$$B_{n,t} = 1 + B_{n-1,t+1}\,\frac{T - t - 1}{T - t}$$

Iterating forward in $n$ from $B_{1,t} = 1$:

$$B_{n,t} = 1 + \frac{T-t-1}{T-t} + \frac{T-t-2}{T-t} + \cdots + \frac{T-t-n+1}{T-t}$$

$$= n - \frac{1}{T-t}\sum_{k=1}^{n-1} k = n - \frac{(n-1)\,n}{2(T-t)}$$

$$\boxed{B_{n,t} = n\!\left(1 - \frac{n-1}{2(T-t)}\right)} \qquad \text{(Eq. 16)}$$

### Case 2: $n > T - t$

When maturity exceeds time remaining, $B$ saturates:

$$\boxed{B_{n,t} = \frac{T - t + 1}{2}} \qquad \text{(Eq. 17)}$$

### Verification

These formulas are numerically verified in `tests/test_model.py::TestBCoefficients` for hundreds of $(n, t)$ pairs.

---

## 7. Yield Spread Formula

The spread at maturity $n$ and time $t$ is (Eq. 19):

$$\delta_t^n = \frac{1}{n}\bigl(A_{n,t} + B_{n,t}\,z_t\bigr)$$

---

## 8. Estimation

### Identification

The 1-month (4-week) spread $\delta_t^4$ is observed without error. The latent factor is extracted as:

$$z_t = B_{4,t}^{-1}\!\bigl(4\,\delta_t^4 - A_{4,t}\bigr)$$

### Measurement equation

For all other maturities $n \in \mathcal{U} = \{13, 26, 52, 104, 156, 208, 260\}$ weeks:

$$\delta_t^n = \frac{1}{n}\bigl(A_{n,t} + B_{n,t}\,z_t\bigr) + \eta_t^n$$

where $\eta_t^n$ are i.i.d. measurement errors with equal variance across maturities.

### Cross-sectional ML estimator

Under the equal-variance assumption, log-likelihood maximisation reduces to (Eq. 20):

$$\hat{\lambda},\;\hat{\sigma} = \arg\min_{\lambda,\,\sigma} \sum_{t=1}^{N} \sum_{n \in \mathcal{U}} \bigl(\eta_t^n\bigr)^2$$

### Implementation

The optimisation works in the transformed space $(\lambda,\;\log\sigma)$ to enforce $\sigma > 0$, using L-BFGS-B (equivalent to MATLAB's `fminunc`). Standard errors are obtained from the inverse of the numerically evaluated Hessian at the natural-scale optimum $(\hat\lambda,\;\hat\sigma)$, using central finite differences with step $\epsilon = 10^{-3}$.

---

## 9. Computational Notes

The double recursion for $A_{n,t}$ requires a specific evaluation order:

1. **$B$ coefficients** can be computed for all $(n, t)$ independently (they do not depend on $\lambda$ or $\sigma$).
2. **$A$ coefficients** must be computed backward in $t$: start from $A_{1,T-1} = 0$, then compute $A_{2,T-2}$, then $A_{2,T-3}$, $A_{3,T-3}$, and so on.
3. For $t \geq T$: $A_{n,t} = B_{n,t} = 0$ for all $n$.

In the implementation, both recursions are evaluated in a single double loop over `tau` (time index) and `n` (maturity index), with `tau` in the outer loop. This is JIT-compiled via Numba for a ~100x speedup over pure Python.
