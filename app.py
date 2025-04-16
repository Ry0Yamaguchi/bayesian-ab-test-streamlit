import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

st.title("Bayesian A/B Test")

st.subheader("Group A")
a_success = st.number_input("Successes (A)", min_value=0, value=50)
a_total = st.number_input("Trials (A)", min_value=1, value=500)

st.subheader("Group B")
b_success = st.number_input("Successes (B)", min_value=0, value=70)
b_total = st.number_input("Trials (B)", min_value=1, value=480)

st.subheader("Prior distribution (Beta)")
alpha_prior = st.number_input("Alpha (α)", min_value=0.1, value=1.0)
beta_prior = st.number_input("Beta (β)", min_value=0.1, value=1.0)

# Posterior parameters
a_alpha_post = a_success + alpha_prior
a_beta_post = a_total - a_success + beta_prior
b_alpha_post = b_success + alpha_prior
b_beta_post = b_total - b_success + beta_prior

# Sampling
samples = 100_000
a_samples = beta.rvs(a_alpha_post, a_beta_post, size=samples)
b_samples = beta.rvs(b_alpha_post, b_beta_post, size=samples)
lift_samples = (b_samples - a_samples) / a_samples

# Metrics
p_b_better = np.mean(b_samples > a_samples)
expected_lift = np.mean(lift_samples)

st.metric("P(B > A)", f"{p_b_better:.2%}")
st.metric("Expected Lift", f"{expected_lift:.2%}")

# Posterior distributions plot
fig1, ax1 = plt.subplots()
x = np.linspace(0, max(max(a_samples), max(b_samples)), 1000)
ax1.plot(x, beta.pdf(x, a_alpha_post, a_beta_post), label="Posterior A")
ax1.plot(x, beta.pdf(x, b_alpha_post, b_beta_post), label="Posterior B")
ax1.set_xlabel("Conversion Rate")
ax1.set_ylabel("Density")
ax1.set_title("Posterior Distributions")
ax1.legend()
st.pyplot(fig1)

# Lift distribution plot
fig2, ax2 = plt.subplots()
ax2.hist(lift_samples, bins=100, density=True, alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', label='Lift = 0')
ax2.set_xlabel("Relative Lift ((B - A) / A)")
ax2.set_ylabel("Density")
ax2.set_title("Lift Distribution")
ax2.legend()
st.pyplot(fig2)
