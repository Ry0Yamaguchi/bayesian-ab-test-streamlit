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

st.subheader("事前分布（Beta）事前のCVRが分からない場合はデフォルトのBe(1,1)でOKです")
alpha_prior = st.number_input("α", min_value=0.1, value=1.0)
beta_prior = st.number_input("β", min_value=0.1, value=1.0)

# 事後分布パラメータ
a_alpha_post = a_success + alpha_prior
a_beta_post = a_total - a_success + beta_prior
b_alpha_post = b_success + alpha_prior
b_beta_post = b_total - b_success + beta_prior

# サンプリング
samples = 100_000
a_samples = beta.rvs(a_alpha_post, a_beta_post, size=samples)
b_samples = beta.rvs(b_alpha_post, b_beta_post, size=samples)
lift_samples = (b_samples - a_samples) / a_samples

# 結果表示
p_b_better = np.mean(b_samples > a_samples)
expected_lift = np.mean(lift_samples)

st.metric("BがAより優れている確率", f"{p_b_better:.2%}")
st.metric("期待リフト率", f"{expected_lift:.2%}")

# 事後分布グラフ（英語ラベル）
fig1, ax1 = plt.subplots()
x = np.linspace(0, max(max(a_samples), max(b_samples)), 1000)
ax1.plot(x, beta.pdf(x, a_alpha_post, a_beta_post), label="Posterior A")
ax1.plot(x, beta.pdf(x, b_alpha_post, b_beta_post), label="Posterior B")
ax1.set_xlabel("Conversion Rate")
ax1.set_ylabel("Density")
ax1.set_title("Posterior Distributions")
ax1.legend()
st.pyplot(fig1)

# リフト分布グラフ（英語ラベル）
fig2, ax2 = plt.subplots()
ax2.hist(lift_samples, bins=100, density=True, alpha=0.7)
ax2.axvline(0, color='red', linestyle='--', label='Lift = 0')
ax2.set_xlabel("Relative Lift ((B - A) / A)")
ax2.set_ylabel("Density")
ax2.set_title("Lift Distribution")
ax2.legend()
st.pyplot(fig2)


