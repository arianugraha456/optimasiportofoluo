import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

st.set_page_config(page_title="Optimizer Portofolio IDX30", layout="wide", initial_sidebar_state="expanded")

idx30 = {
    'BBCA.JK': 'Bank Central Asia',
    'ASII.JK': 'Astra International',
    'TLKM.JK': 'Telkom Indonesia',
    'UNVR.JK': 'Unilever Indonesia',
    'BMRI.JK': 'Bank Mandiri',
    'HMSP.JK': 'HM Sampoerna',
    'ICBP.JK': 'Indofood CBP',
    'SMGR.JK': 'Semen Indonesia',
    'MNCN.JK': 'MNC Studios',
    'INDF.JK': 'Indofood Sukses Makmur',
    'BBNI.JK': 'Bank Negara Indonesia',
    'PTBA.JK': 'Bukit Asam',
    'ADRO.JK': 'Adaro Energy',
    'BRIS.JK': 'Bank Syariah Indonesia',
    'EXCL.JK': 'XL Axiata',
    'INCO.JK': 'Vale Indonesia',
    'BSDE.JK': 'Bumi Serpong Damai',
    'CTRA.JK': 'Ciputra Development',
    'JSMR.JK': 'Jasa Marga',
    'MEDC.JK': 'Medco Energi',
    'PGAS.JK': 'Perusahaan Gas Negara',
    'SCMA.JK': 'Surya Citra Media',
    'WIKA.JK': 'Wijaya Karya',
    'AKRA.JK': 'AKR Corporindo',
    'MYOR.JK': 'Mayora Indah',
    'CPIN.JK': 'Charoen Pokphand Indonesia',
    'KLBF.JK': 'Kalbe Farma',
    'SRIL.JK': 'Sri Rejeki Isman',
    'TINS.JK': 'Timah',
    'GGRM.JK': 'Gudang Garam',
}

def portfolio_variance(weights, cov):
    return weights.T @ cov @ weights

def optimize_mvep(returns_df):
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov().values
    n = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.random.dirichlet(np.ones(n), size=1)[0]
    result = minimize(portfolio_variance, init_guess, args=(cov_matrix,), method='SLSQP',
                      bounds=bounds, constraints=constraints)
    if not result.success:
        st.error("Optimasi gagal: " + result.message)
    return result.x, mean_returns.values, cov_matrix

def calculate_sharpe_ratio(returns_df, weights, risk_free_rate=0.0):
    portfolio_returns = returns_df @ weights
    excess_returns = portfolio_returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

def calculate_var_historical(returns_series, confidence_level=0.95):
    return np.percentile(returns_series, (1 - confidence_level) * 100)

def calculate_var_parametric(returns_series, confidence_level=0.95):
    mean = np.mean(returns_series)
    std_dev = np.std(returns_series)
    return norm.ppf(1 - confidence_level, mean, std_dev)

def calculate_var_monte_carlo(returns_series, confidence_level=0.95, num_simulations=10000):
    mean = np.mean(returns_series)
    std_dev = np.std(returns_series)
    simulated_returns = np.random.normal(mean, std_dev, num_simulations)
    return np.percentile(simulated_returns, (1 - confidence_level) * 100)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

body {
    font-family: 'Inter', sans-serif;
    background-color: #ffffff;
    color: #6b7280;
    margin: 0;
    padding: 0;
}

.main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 3rem 2rem 4rem 2rem;
}

h1.main-title {
    font-weight: 700;
    font-size: 3.5rem;
    color: #111827;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}

h3.sub-title {
    font-weight: 600;
    font-size: 1.5rem;
    color: #6b7280;
    margin-bottom: 2.5rem;
}

.card {
    background: #f9fafb;
    padding: 2rem 2.5rem;
    border-radius: 0.75rem;
    box-shadow: 0 0 10px rgb(0 0 0 / 0.05);
    margin-bottom: 2.5rem;
}

h4.section-title {
    color: #2563eb;
    font-weight: 600;
    font-size: 1.25rem;
    margin-bottom: 1rem;
}

.output-section {
    margin-top: 1.5rem;
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
}

.bubble {
    flex: 1 1 220px;
    background-color: #e0f2fe;
    color: #0369a1;
    border-radius: 1rem;
    padding: 1rem 1.5rem;
    font-size: 1.1rem;
    font-weight: 600;
    box-shadow: 0 1px 5px rgb(0 0 0 / 0.08);
    text-align: center;
    user-select:none;
}

input, select, .stNumberInput>div>input {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500;
    font-size: 1rem !important;
    padding: 0.5rem 0.75rem !important;
    border-radius: 0.5rem !important;
    border: 1px solid #d1d5db !important;
    color: #374151 !important;
}

.stButton > button {
    background-color: #2563eb;
    color: white;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    border-radius: 0.5rem;
    border: none;
    transition: background-color 0.3s ease;
    font-size: 1rem;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    cursor: pointer;
}

.stMultiselect>div>div>div>input {
    font-weight: 500;
    font-size: 1rem;
    font-family: 'Inter', sans-serif !important;
}

.chart-subheader {
    font-weight: 600;
    font-size: 1.2rem;
    margin-bottom: 0.75rem;
    color: #374151;
}

.interpretation {
    background: #f3f4f6;
    border-left: 4px solid #2563eb;
    padding: 1rem 1.5rem;
    border-radius: 0.5rem;
    margin-top: -1.5rem;
    margin-bottom: 2.5rem;
    color: #374151;
    font-size: 1rem;
    line-height: 1.5;
}

.interpretation h5 {
    font-weight: 700;
    color: #2563eb;
    margin-bottom: 0.5rem;
}

@media (max-width: 768px) {
    .output-section {
        flex-direction: column;
        gap: 0.75rem;
    }
}
</style>
""", unsafe_allow_html=True)

# Sidebar for Usage Guide
with st.sidebar:
    st.markdown("---")
    st.markdown('<h4 class="section-title">Panduan Penggunaan</h4>', unsafe_allow_html=True)
    st.markdown("""
    <ul style="color:#374151; font-size:0.9rem; line-height:1.6; padding-left:1rem;">
        <li>Pilih minimal 2 saham IDX30 yang ingin Anda analisis dari daftar di bawah.</li>
        <li>Masukkan jumlah modal investasi Anda dalam Rupiah (Rp).</li>
        <li>Sesuaikan tingkat kepercayaan untuk Value at Risk (VaR), dengan 95% sebagai rekomendasi umum.</li>
        <li>Klik tombol **Hitung Optimasi** untuk memulai analisis dan melihat hasil portofolio yang dioptimalkan.</li>
        <li>Hasil analisis, metrik kinerja, dan visualisasi akan ditampilkan di bagian utama aplikasi.</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("---")


st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<h1 class="main-title">Optimizer Portofolio IDX30 & Analisis Visual Portofolio</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="sub-title">Analisis dan optimalkan portofolio Anda dengan wawasan visual elegan dari saham IDX30 — sederhana, ramping, dan kuat.</h3>', unsafe_allow_html=True)

# Removed the usage guide card from the main section

with st.form("input_form", clear_on_submit=False):
    stock_options = [f"{ticker} - {name}" for ticker, name in idx30.items()]
    selected = st.multiselect(
        "Pilih saham IDX30 (minimal 2):",
        options=stock_options,
        help="Cari dan pilih saham yang ingin dianalisis"
    )
    modal = st.number_input(
        "Modal Investasi (Rp):",
        min_value=100000,
        value=10000000,
        step=100000,
        format="%d"
    )
    confidence_level = st.select_slider(
        "Pilih Tingkat Kepercayaan untuk Value at Risk (VaR):",
        options=[0.90, 0.95, 0.99],
        value=0.95
    )
    submit_button = st.form_submit_button(label="Hitung Optimasi")

if submit_button:
    if not selected or len(selected) < 2:
        st.warning("⚠ Silakan pilih minimal 2 saham untuk optimasi portofolio.")
    else:
        selected_tickers = [s.split(" - ")[0] for s in selected]
        with st.spinner("Mengambil data dan melakukan optimasi portofolio..."):
            data = yf.download(selected_tickers, period="3y", interval="1d")['Close'].dropna()
            returns = np.log(data / data.shift(1)).dropna()

            weights, mean_returns, cov_matrix = optimize_mvep(returns)
            port_return = np.sum(weights * mean_returns)
            sharpe_ratio = calculate_sharpe_ratio(returns, weights)
            port_variance = portfolio_variance(weights, cov_matrix)
            port_returns_series = returns @ weights

            var_historical = calculate_var_historical(port_returns_series, confidence_level)
            var_parametric = calculate_var_parametric(port_returns_series, confidence_level)
            var_monte_carlo = calculate_var_monte_carlo(port_returns_series, confidence_level)

            var_historical_percent = var_historical * 100
            var_parametric_percent = var_parametric * 100
            var_monte_carlo_percent = var_monte_carlo * 100

            portfolio = pd.DataFrame({
                'Saham': [idx30[t] for t in selected_tickers],
                'Ticker': selected_tickers,
                'Bobot (%)': weights * 100,
                'Investasi (Rp)': weights * modal
            })
            portfolio.reset_index(drop=True, inplace=True)
            portfolio.index += 1

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Bobot Portofolio yang Dioptimalkan")
        st.dataframe(
            portfolio.style.format({
                "Bobot (%)": "{:.2f}",
                "Investasi (Rp)": "Rp {:,.0f}"
            }),
            use_container_width=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="output-section">', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble">Perkiraan Pengembalian Harian: {port_return*100:.4f}%</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble">Rasio Sharpe: {sharpe_ratio:.4f}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble">Varians Portofolio: {port_variance:.4f}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<h4 class="section-title">Value at Risk (VaR)</h4>', unsafe_allow_html=True)
        st.markdown('<div class="output-section">', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble">1. VaR Historis pada tingkat kepercayaan {confidence_level*100:.0f}%: Rp {abs(var_historical * modal):,.0f} ({var_historical_percent:.2f}%)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble">2. VaR Parametrik pada tingkat kepercayaan {confidence_level*100:.0f}%: Rp {abs(var_parametric * modal):,.0f} ({var_parametric_percent:.2f}%)</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bubble">3. VaR Monte Carlo pada tingkat kepercayaan {confidence_level*100:.0f}%: Rp {abs(var_monte_carlo * modal):,.0f} ({var_monte_carlo_percent:.2f}%)</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<h4 class="section-title">Interpretasi Value at Risk (VaR)</h4>', unsafe_allow_html=True)
        st.markdown(f"""
        <ul>
            <li><strong>VaR Historis</strong>: Mengukur potensi kerugian maksimum yang dapat terjadi dalam periode tertentu berdasarkan data historis. Misalnya, jika VaR Historis adalah Rp {abs(var_historical * modal):,.0f} ({var_historical_percent:.2f}%), berarti ada kemungkinan {100 - confidence_level * 100:.0f}% bahwa kerugian akan melebihi jumlah tersebut dalam satu hari.</li>
            <li><strong>VaR Parametrik</strong>: Menggunakan asumsi distribusi normal untuk menghitung potensi kerugian. Jika VaR Parametrik adalah Rp {abs(var_parametric * modal):,.0f} ({var_parametric_percent:.2f}%), ini menunjukkan bahwa ada kemungkinan {100 - confidence_level * 100:.0f}% bahwa kerugian akan melebihi jumlah tersebut dalam satu hari.</li>
            <li><strong>VaR Monte Carlo</strong>: Menggunakan simulasi untuk memperkirakan potensi kerugian dengan mempertimbangkan berbagai skenario pasar. Jika VaR Monte Carlo adalah Rp {abs(var_monte_carlo * modal):,.0f} ({var_monte_carlo_percent:.2f}%), ini menunjukkan bahwa ada kemungkinan {100 - confidence_level * 100:.0f}% bahwa kerugian akan melebihi jumlah tersebut dalam satu hari.</li>
        </ul>
        """, unsafe_allow_html=True)

        # Grafik Distribusi Bobot Portofolio
        st.subheader("Distribusi Bobot Portofolio")
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        colors = plt.get_cmap('tab20').colors
        portfolio.set_index('Saham')['Bobot (%)'].plot.pie(autopct='%1.1f%%', startangle=90,
                                                           colors=colors[:len(portfolio)], ax=ax1)
        ax1.set_ylabel("")
        st.pyplot(fig1)
        st.markdown("""
        <div class="interpretation">
            <h5>Interpretasi Distribusi Bobot Portofolio</h5>
            Grafik ini menunjukkan proporsi investasi pada masing-masing saham di dalam portofolio Anda. Anda dapat melihat saham mana yang mendapatkan porsi alokasi terbesar yang memengaruhi performa dan risiko portofolio.
        </div>
        """, unsafe_allow_html=True)

        # Grafik Distribusi Pengembalian Harian Portofolio
        st.subheader("Distribusi Pengembalian Harian Portofolio")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.hist(port_returns_series, bins=50, color='#2563eb', alpha=0.75)
        ax2.set_xlabel("Pengembalian Harian")
        ax2.set_ylabel("Frekuensi")
        ax2.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig2)
        st.markdown("""
        <div class="interpretation">
            <h5>Interpretasi Distribusi Pengembalian Harian Portofolio</h5>
            Histogram ini memperlihatkan frekuensi pengembalian harian portofolio. Distribusi yang simetris menunjukkan kestabilan pengembalian, sementara sebaran yang lebar atau adanya outlier mengindikasikan volatilitas dan risiko yang lebih tinggi.
        </div>
        """, unsafe_allow_html=True)

        # Grafik Pengembalian Kumulatif Portofolio
        st.subheader("Pengembalian Kumulatif Portofolio")
        cumulative_returns = (1 + port_returns_series).cumprod() - 1
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.plot(cumulative_returns, color='#111827', linewidth=2.5)
        ax3.set_xlabel("Hari")
        ax3.set_ylabel("Pengembalian Kumulatif")
        ax3.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig3)
        st.markdown("""
        <div class="interpretation">
            <h5>Interpretasi Pengembalian Kumulatif Portofolio</h5>
            Grafik garis ini menampilkan perkembangan nilai portofolio secara kumulatif dari waktu ke waktu. Tren naik mengindikasikan return positif yang konsisten, dan tren turun menunjukkan kerugian kumulatif.
        </div>
        """, unsafe_allow_html=True)

        # Grafik Volatilitas 30-hari yang Menggulung
        st.subheader("Volatilitas 30-hari yang Menggulung")
        rolling_volatility = port_returns_series.rolling(window=30).std() * np.sqrt(252)
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        ax4.plot(rolling_volatility, color='#d97706', linewidth=2.5)
        ax4.set_xlabel("Hari")
        ax4.set_ylabel("Volatilitas")
        ax4.grid(True, linestyle='--', alpha=0.4)
        st.pyplot(fig4)
        st.markdown("""
        <div class="interpretation">
            <h5>Interpretasi Volatilitas 30-hari yang Menggulung</h5>
            Grafik ini menunjukkan fluktuasi risiko portofolio dalam periode rolling 30 hari. Nilai volatilitas yang tinggi mengindikasikan risiko besar, sedangkan nilai rendah menandakan kestabilan.
        </div>
        """, unsafe_allow_html=True)

        # Grafik Heatmap Korelasi Antara Saham yang Dipilih
        st.subheader("Heatmap Korelasi Antara Saham yang Dipilih")
        correlation_matrix = returns.corr()
        fig5, ax5 = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax5,
                    cbar_kws={"shrink": 0.7}, square=True, linewidths=0.7)
        st.pyplot(fig5)
        st.markdown("""
        <div class="interpretation">
            <h5>Interpretasi Heatmap Korelasi Antara Saham yang Dipilih</h5>
            Heatmap ini menampilkan tingkat korelasi antar saham. Korelasi positif tinggi menunjukkan saham bergerak bersama, sedangkan korelasi negatif berarti pergerakan berlawanan. Diversifikasi portofolio dapat dicapai dengan memilih saham berkorelasi rendah atau negatif.
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("Silakan pilih saham, masukkan modal, dan klik 'Hitung Optimasi' untuk memulai analisis.")

st.markdown("</div>", unsafe_allow_html=True)