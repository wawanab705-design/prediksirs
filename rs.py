import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Analisis Rumah Sakit",
    page_icon="🏥",
    layout="wide"
)

# Fungsi untuk memuat data dari file lokal
@st.cache_data
def load_data_from_files():
    try:
        # Membaca file CSV langsung dengan pandas
        df_pribadi = pd.read_csv("belanja-jan-nov2025.csv", delimiter=";", encoding='utf-8')
        df_asuransi = pd.read_csv("belanja-pasien-asuransi2025.csv", delimiter=";", encoding='utf-8')
    except:
        # Fallback jika delimiter utama gagal
        try:
            df_pribadi = pd.read_csv("belanja-jan-nov2025.csv", delimiter=",", encoding='utf-8')
            df_asuransi = pd.read_csv("belanja-pasien-asuransi2025.csv", delimiter=",", encoding='utf-8')
        except Exception as e:
            st.error(f"Error membaca file CSV: {e}")
            return pd.DataFrame()

    # Standarisasi nama kolom
    columns = ["NO", "RM", "EPS", "NAMA", "ADMISI", "DOKTER", "RAWAT", 
               "JENIS_PELAYANAN", "PENJAMIN", "TOTAL", "DISKON", "MENINGGAL"]
    
    if len(df_pribadi.columns) == len(columns):
        df_pribadi.columns = columns
    if len(df_asuransi.columns) == len(columns):
        df_asuransi.columns = columns
    
    # Tambahkan kolom jenis pasien
    df_pribadi["jenis_pasien"] = "Pribadi"
    df_asuransi["jenis_pasien"] = "Asuransi"
    
    # Gabungkan data
    df = pd.concat([df_pribadi, df_asuransi], ignore_index=True)
    
    # Preprocessing
    # Bersihkan format TOTAL
    df["TOTAL"] = df["TOTAL"].astype(str).str.replace(',', '').str.replace('"', '').str.strip()
    
    # Konversi ke numeric
    df["biaya_clean"] = pd.to_numeric(df["TOTAL"], errors='coerce')
    
    # Parse tanggal
    df["ADMISI"] = df["ADMISI"].astype(str).str.replace('\u00A0', ' ', regex=False)
    
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, '%d/%m/%Y %H:%M')
        except:
            try:
                return datetime.strptime(date_str, '%d/%m/%Y')
            except:
                return pd.NaT
    
    df["datetime"] = df["ADMISI"].apply(parse_date)
    
    # Ekstrak bulan dan tahun
    df["bulan"] = df["datetime"].dt.month
    df["tahun"] = df["datetime"].dt.year
    
    # Filter hanya 2025
    df = df[df["tahun"] == 2025]
    
    return df

# Fungsi untuk perhitungan KPI
def calculate_kpis(df):
    if df.empty:
        return {
            "total_pendapatan": 0,
            "jumlah_kunjungan": 0,
            "rasio_asuransi_pribadi": 0,
            "pertumbuhan_bulanan": 0,
            "asuransi_count": 0,
            "pribadi_count": 0
        }
    
    total_pendapatan = df["biaya_clean"].sum()
    jumlah_kunjungan = len(df)
    
    asuransi_count = len(df[df["jenis_pasien"] == "Asuransi"])
    pribadi_count = len(df[df["jenis_pasien"] == "Pribadi"])
    
    if pribadi_count > 0:
        rasio_asuransi_pribadi = asuransi_count / pribadi_count
    else:
        rasio_asuransi_pribadi = 0
    
    # Hitung pertumbuhan bulanan
    monthly_revenue = df.groupby("bulan")["biaya_clean"].sum().reset_index()
    
    if len(monthly_revenue) > 1:
        monthly_revenue = monthly_revenue.sort_values("bulan")
        first_rev = monthly_revenue["biaya_clean"].iloc[0]
        last_rev = monthly_revenue["biaya_clean"].iloc[-1]
        
        if first_rev > 0:
            pertumbuhan_bulanan = ((last_rev - first_rev) / first_rev) * 100
        else:
            pertumbuhan_bulanan = 0
    else:
        pertumbuhan_bulanan = 0
    
    return {
        "total_pendapatan": total_pendapatan,
        "jumlah_kunjungan": jumlah_kunjungan,
        "rasio_asuransi_pribadi": rasio_asuransi_pribadi,
        "pertumbuhan_bulanan": pertumbuhan_bulanan,
        "asuransi_count": asuransi_count,
        "pribadi_count": pribadi_count
    }

# Fungsi prediksi sederhana
def predict_revenue_2026(df):
    if df.empty:
        return pd.DataFrame({"bulan": range(1, 13), "prediction": [0]*12})
    
    monthly_rev = df.groupby("bulan")["biaya_clean"].sum().reset_index()
    
    if len(monthly_rev) < 2:
        # Jika data kurang, gunakan rata-rata
        avg_revenue = monthly_rev["biaya_clean"].mean() if not monthly_rev.empty else 0
        predictions = [avg_revenue * 1.1] * 12  # 10% growth assumption
    else:
        # Simple linear regression manual
        X = monthly_rev["bulan"].values
        y = monthly_rev["biaya_clean"].values
        
        # Calculate slope and intercept
        n = len(X)
        mean_x = np.mean(X)
        mean_y = np.mean(y)
        
        numerator = np.sum((X - mean_x) * (y - mean_y))
        denominator = np.sum((X - mean_x) ** 2)
        
        if denominator != 0:
            slope = numerator / denominator
            intercept = mean_y - slope * mean_x
            
            # Predict for 2026 (months 1-12)
            predictions = [slope * month + intercept for month in range(1, 13)]
        else:
            avg_revenue = mean_y
            predictions = [avg_revenue * 1.1] * 12
    
    return pd.DataFrame({"bulan": range(1, 13), "prediction": predictions})

# Fungsi format currency
def format_currency(value):
    if value >= 1_000_000_000:
        return f"Rp {value/1_000_000_000:.2f}M"
    elif value >= 1_000_000:
        return f"Rp {value/1_000_000:.2f}Jt"
    elif value >= 1_000:
        return f"Rp {value/1_000:.1f}K"
    else:
        return f"Rp {value:.0f}"

# Main aplikasi
def main():
    st.title("🏥 Dashboard Analisis Rumah Sakit")
    st.markdown("Dashboard ini membantu manajemen mengalokasikan SDM dan mengembangkan fasilitas berdasarkan data nyata.")
    
    # Load data
    with st.spinner("Memuat data..."):
        df = load_data_from_files()
    
    if df.empty:
        st.error("Tidak dapat memuat data. Pastikan file CSV tersedia.")
        return
    
    # Sidebar untuk filter
    st.sidebar.header("⚙️ Filter Data")
    
    # Filter bulan
    bulan_options = sorted(df["bulan"].dropna().unique())
    selected_bulan = st.sidebar.multiselect(
        "Pilih Bulan:",
        bulan_options,
        default=bulan_options
    )
    
    # Filter jenis pasien
    jenis_pasien_options = ["Semua", "Asuransi", "Pribadi"]
    selected_jenis_pasien = st.sidebar.selectbox(
        "Pilih Jenis Pasien:",
        jenis_pasien_options
    )
    
    # Filter poliklinik
    poliklinik_options = ["Semua"] + sorted(df["JENIS_PELAYANAN"].dropna().unique().tolist())
    selected_poliklinik = st.sidebar.selectbox(
        "Pilih Poliklinik:",
        poliklinik_options
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_bulan:
        filtered_df = filtered_df[filtered_df["bulan"].isin(selected_bulan)]
    
    if selected_jenis_pasien != "Semua":
        filtered_df = filtered_df[filtered_df["jenis_pasien"] == selected_jenis_pasien]
    
    if selected_poliklinik != "Semua":
        filtered_df = filtered_df[filtered_df["JENIS_PELAYANAN"] == selected_poliklinik]
    
    # Hitung KPI
    kpis = calculate_kpis(filtered_df)
    
    # A. Ringkasan Kinerja
    st.header("📊 Ringkasan Kinerja")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Pendapatan 2025",
            value=format_currency(kpis["total_pendapatan"]),
            delta=f"{kpis['pertumbuhan_bulanan']:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Jumlah Kunjungan",
            value=f"{kpis['jumlah_kunjungan']:,}"
        )
    
    with col3:
        st.metric(
            label="Rasio Asuransi : Pribadi",
            value=f"{kpis['rasio_asuransi_pribadi']:.2f}",
            help=f"Asuransi: {kpis['asuransi_count']:,} | Pribadi: {kpis['pribadi_count']:,}"
        )
    
    with col4:
        color = "green" if kpis['pertumbuhan_bulanan'] > 0 else "red"
        st.markdown(f"""
        <div style="background-color:{color}20; padding:10px; border-radius:10px; border-left:5px solid {color}">
            <h4 style="margin:0; color:{color}">Pertumbuhan Bulanan</h4>
            <h2 style="margin:0; color:{color}">{kpis['pertumbuhan_bulanan']:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # B. Grafik 1: Tren Pendapatan
    st.header("📈 Tren Pendapatan Bulanan")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Data bulanan
        monthly_rev = filtered_df.groupby(["bulan", "jenis_pasien"]).agg({
            "biaya_clean": "sum",
            "NAMA": "count"
        }).reset_index()
        monthly_rev.columns = ["bulan", "jenis_pasien", "pendapatan", "kunjungan"]
        
        # Prediksi 2026
        predictions = predict_revenue_2026(filtered_df)
        
        # Buat grafik
        fig = go.Figure()
        
        # Tambahkan garis untuk setiap jenis pasien
        for jenis in monthly_rev["jenis_pasien"].unique():
            data_jenis = monthly_rev[monthly_rev["jenis_pasien"] == jenis]
            fig.add_trace(go.Scatter(
                x=data_jenis["bulan"],
                y=data_jenis["pendapatan"],
                mode="lines+markers",
                name=f"2025 {jenis}",
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        # Tambahkan prediksi 2026
        fig.add_trace(go.Scatter(
            x=predictions["bulan"],
            y=predictions["prediction"],
            mode="lines",
            name="Prediksi 2026",
            line=dict(dash="dash", width=2, color="red"),
            opacity=0.7
        ))
        
        fig.update_layout(
            title="Tren Pendapatan Bulanan (2025) dan Prediksi 2026",
            xaxis_title="Bulan",
            yaxis_title="Pendapatan (Rp)",
            hovermode="x unified",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Statistik")
        st.metric("Total 2025", format_currency(kpis["total_pendapatan"]))
        
        if not monthly_rev.empty:
            avg_monthly = monthly_rev["pendapatan"].mean()
            st.metric("Rata-rata/Bulan", format_currency(avg_monthly))
        
        st.subheader("🎯 Prediksi 2026")
        if not predictions.empty:
            pred_total = predictions["prediction"].sum()
            st.metric("Total Prediksi", format_currency(pred_total))
    
    st.markdown("---")
    
    # C. Grafik 2: Top 10 Poliklinik
    st.header("🏆 Top 10 Poliklinik")
    
    if not filtered_df.empty:
        top_poliklinik = filtered_df.groupby("JENIS_PELAYANAN").agg({
            "biaya_clean": "sum",
            "NAMA": "count"
        }).reset_index()
        top_poliklinik.columns = ["Poliklinik", "Pendapatan", "Kunjungan"]
        top_poliklinik = top_poliklinik.sort_values("Pendapatan", ascending=False).head(10)
        
        # Buat bar chart
        fig2 = go.Figure()
        
        fig2.add_trace(go.Bar(
            y=top_poliklinik["Poliklinik"],
            x=top_poliklinik["Pendapatan"],
            name="Pendapatan",
            orientation="h",
            marker_color="#1f77b4",
            hovertemplate="%{y}<br>Pendapatan: Rp %{x:,.0f}<extra></extra>"
        ))
        
        fig2.add_trace(go.Scatter(
            y=top_poliklinik["Poliklinik"],
            x=top_poliklinik["Kunjungan"] * (top_poliklinik["Pendapatan"].max() / top_poliklinik["Kunjungan"].max() * 0.3),
            name="Kunjungan (skala berbeda)",
            mode="markers+text",
            marker=dict(size=10, color="#ff7f0e"),
            text=top_poliklinik["Kunjungan"],
            textposition="middle right",
            hovertemplate="%{y}<br>Kunjungan: %{text}<extra></extra>"
        ))
        
        fig2.update_layout(
            title="Pendapatan dan Jumlah Kunjungan per Poliklinik",
            xaxis_title="Pendapatan (Rp)",
            yaxis_title="Poliklinik",
            height=500,
            template="plotly_white",
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Tidak ada data untuk filter yang dipilih.")
    
    st.markdown("---")
    
    # D. Grafik 3: Kontribusi Penjamin
    st.header("💰 Kontribusi Penjamin")
    
    if not filtered_df.empty:
        penjamin_contribution = filtered_df.groupby(["JENIS_PELAYANAN", "jenis_pasien"]).agg({
            "biaya_clean": "sum"
        }).reset_index()
        penjamin_contribution.columns = ["Poliklinik", "Jenis Pasien", "Pendapatan"]
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Stacked bar chart
            fig3 = px.bar(
                penjamin_contribution,
                x="Poliklinik",
                y="Pendapatan",
                color="Jenis Pasien",
                title="Kontribusi Pendapatan per Poliklinik",
                labels={"Pendapatan": "Pendapatan (Rp)", "Jenis Pasien": "Jenis Pasien"},
                template="plotly_white",
                height=400
            )
            
            fig3.update_layout(
                xaxis_tickangle=-45,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            # Pie chart
            total_by_type = filtered_df.groupby("jenis_pasien")["biaya_clean"].sum().reset_index()
            
            if not total_by_type.empty:
                fig4 = px.pie(
                    total_by_type,
                    values="biaya_clean",
                    names="jenis_pasien",
                    title="Distribusi Pendapatan",
                    hole=0.4,
                    height=300
                )
                
                fig4.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig4, use_container_width=True)
                
                # Persentase
                st.subheader("Persentase")
                for _, row in total_by_type.iterrows():
                    percentage = (row["biaya_clean"] / total_by_type["biaya_clean"].sum()) * 100
                    st.metric(f"{row['jenis_pasien']}", f"{percentage:.1f}%")
    
    st.markdown("---")
    
    # E. Rekomendasi
    st.header("🎯 Rekomendasi Manajemen")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👨‍⚕️ Alokasi SDM")
        if not filtered_df.empty and "JENIS_PELAYANAN" in filtered_df.columns:
            top_poliklinik_list = filtered_df.groupby("JENIS_PELAYANAN")["biaya_clean"].sum().nlargest(3)
            st.info(f"""
            **Prioritas:**
            
            1. {top_poliklinik_list.index[0] if len(top_poliklinik_list) > 0 else 'N/A'}
            2. {top_poliklinik_list.index[1] if len(top_poliklinik_list) > 1 else 'N/A'}
            3. {top_poliklinik_list.index[2] if len(top_poliklinik_list) > 2 else 'N/A'}
            """)
    
    with col2:
        st.subheader("🏗️ Pengembangan Fasilitas")
        st.info("""
        **Rekomendasi:**
        
        • Investasi peralatan medis modern
        • Digitalisasi sistem rekam medis
        • Pengembangan layanan telemedisin
        """)
    
    with col3:
        st.subheader("📈 Strategi Bisnis")
        st.info(f"""
        **Analisis Rasio: {kpis['rasio_asuransi_pribadi']:.2f}**
        
        • {'Tingkatkan kerjasama asuransi' if kpis['rasio_asuransi_pribadi'] < 1 else 'Optimalkan layanan pribadi'}
        • Kembangkan paket layanan premium
        • Sistem pembayaran fleksibel
        """)
    
    # Tampilkan data detail
    st.markdown("---")
    st.header("📋 Data Detail")
    
    if st.checkbox("Tampilkan Preview Data"):
        st.dataframe(
            filtered_df[["NAMA", "JENIS_PELAYANAN", "jenis_pasien", "bulan", "biaya_clean", "DOKTER"]].head(50),
            use_container_width=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: gray;">
            <p>Dashboard Analisis Rumah Sakit • Data Jan-Nov 2025</p>
            <p>Terakhir diperbarui: Desember 2024</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
