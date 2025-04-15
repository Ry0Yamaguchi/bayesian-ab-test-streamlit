
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

st.title("ベイズABテスト")

st.subheader("グループA")
a_success = st.number_input("成功数 A", min_value=0, value=50)
a_total = st.number_input("試行数 A", min_value=1, value=500)

st.subheader("グループB")
b_success = st.number_input("成功数 B", min_value=0, value=70)
b_total = st.number_input("試行数 B", min_value=1, value=480)

st.subheader("事前分布（Beta）")
alpha_prior = st.number_input("α", min_value=0.1, value=1.0)
beta_prior = st.number_input("β", min_value=0.1, value=1.0)

a_alpha_post = a_success + alpha_prior
a_beta_post = a_total - a_success + beta_prior
b_alpha_post = b_success + alpha_prior
b_beta_post = b_total - b_success + beta_prior

samples = 100_000
a_samples = beta.rvs(a_alpha_post, a_beta_post, size=samples)
b_samples = beta.rvs(b_alpha_post, b_beta_post, size=samples)

p_b_better = np.mean(b_samples > a_samples)
expected_lift = np.mean((b_samples - a_samples) / a_samples)

st.metric("BがAより優れている確率", f"{p_b_better:.2%}")
st.metric("期待リフト率", f"{expected_lift:.2%}")

fig, ax = plt.subplots()
x = np.linspace(0, 0.3, 1000)
ax.plot(x, beta.pdf(x, a_alpha_post, a_beta_post), label="Aの事後分布")
ax.plot(x, beta.pdf(x, b_alpha_post, b_beta_post), label="Bの事後分布")
ax.set_xlabel("コンバージョン率")
ax.set_ylabel("確率密度")
ax.legend()
st.pyplot(fig)
