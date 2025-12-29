# prediksi_belanja_pasien_pyspark.py
import streamlit as st
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Impor PySpark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum as spark_sum, collect_list, udf
from pyspark.sql.types import IntegerType, DoubleType, StringType
import pyspark.sql.functions as F

# Inisialisasi Spark Session
spark = SparkSession.builder \
    .appName("PrediksiBelanjaPasien") \
    .getOrCreate()

GITHUB_CSV_URL = "https://raw.githubusercontent.com/wawanab705-design/asuransips/refs/heads/wawanab705-design-patch-1/pasien-asuransi.csv"

# === 1. LOAD & PREPROCESS DATA ===
@st.cache_data
def load_and_preprocess():
    # Load data menggunakan PySpark
    df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "false") \
        .csv(GITHUB_CSV_URL)
    
    # Simpan nama kolom asli untuk referensi
    original_columns = df.columns
    
    # Ganti nilai tidak valid dengan null
    invalid_values = ["", "*", "**"]
    for column in original_columns:
        df = df.withColumn(
            column,
            when(col(column).isin(invalid_values), None).otherwise(col(column))
        )
    
    # Daftar bulan
    bulan_list = ["JANUARI", "FEBRUARI", "MARET", "APRIL", "MEI", "JUNI",
                  "JULI", "AGUSTUS", "SEPTEMBER", "OKTOBER", "NOVEMBER"]
    pasien_cols = [f"PASIEN {b}" for b in bulan_list]
    omset_cols = [f"OMSET {b}" for b in bulan_list]
    
    # Konversi tipe data untuk kolom numerik
    for col_name in pasien_cols + omset_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast(DoubleType()))
    
    # Ganti null dengan 0 untuk kolom numerik
    for col_name in pasien_cols + omset_cols:
        if col_name in df.columns:
            df = df.withColumn(col_name, F.coalesce(col(col_name), F.lit(0)))
    
    # Ambil opsi unik untuk dropdown
    jenis_df = df.select("JENIS JAMINAN").distinct().filter(col("JENIS JAMINAN").isNotNull())
    nama_df = df.select("NAMA PENJAMIN").distinct().filter(col("NAMA PENJAMIN").isNotNull())
    
    jenis_options = [row["JENIS JAMINAN"] for row in jenis_df.collect()]
    nama_options = [row["NAMA PENJAMIN"] for row in nama_df.collect()]
    
    jenis_options.sort()
    nama_options.sort()
    
    return df, bulan_list, pasien_cols, omset_cols, jenis_options, nama_options

# === 2. STREAMLIT APP ===
def main():
    st.set_page_config(page_title="Kontribusi Asuransi/Perusahaan ke Rumah Sakit", layout="wide")
    st.title("🩺 Kontribusi Asuransi/Perusahaan ke Rumah Sakit")
    st.markdown("Pilih kriteria untuk melihat **tren jumlah pasien dan omset**.")
    
    # Load data dengan PySpark
    df, bulan_list, pasien_cols, omset_cols, jenis_options, nama_options = load_and_preprocess()
    
    # Input form
    jenis = st.selectbox("Jenis Asuransi", options=jenis_options)
    nama = st.multiselect("Nama Asuransi", options=nama_options, default=[])
    
    # Range bulan (1–11 karena data hanya sampai November)
    bulan_start, bulan_end = st.slider(
        "Pilih Rentang Bulan (1 = Januari, 11 = November)",
        min_value=1, max_value=11, value=(1, 6)
    )
    
    # Filter data menggunakan PySpark
    df_filtered = df.filter(col("JENIS JAMINAN") == jenis)
    
    if nama:
        df_filtered = df_filtered.filter(col("NAMA PENJAMIN").isin(nama))
    
    # Hitung jumlah baris untuk memeriksa apakah ada data
    row_count = df_filtered.count()
    
    if row_count == 0:
        st.warning("⚠️ Tidak ada data yang cocok dengan kriteria Anda.")
        st.stop()
    
    # Ambil kolom dalam rentang
    selected_months = bulan_list[bulan_start-1:bulan_end]
    selected_pasien_cols = pasien_cols[bulan_start-1:bulan_end]
    selected_omset_cols = omset_cols[bulan_start-1:bulan_end]
    
    # Pastikan kolom yang dipilih ada dalam DataFrame
    available_pasien_cols = [col for col in selected_pasien_cols if col in df_filtered.columns]
    available_omset_cols = [col for col in selected_omset_cols if col in df_filtered.columns]
    
    if not available_pasien_cols or not available_omset_cols:
        st.error("Kolom yang diperlukan tidak ditemukan dalam dataset.")
        st.stop()
    
    # Agregasi total per bulan menggunakan PySpark
    # Pertama, hitung total untuk setiap kolom
    aggregation_exprs = []
    for col_name in available_pasien_cols + available_omset_cols:
        aggregation_exprs.append(spark_sum(col_name).alias(col_name))
    
    aggregated_df = df_filtered.agg(*aggregation_exprs)
    
    # Konversi hasil ke dictionary
    aggregated_row = aggregated_df.collect()[0]
    aggregated_dict = aggregated_row.asDict()
    
    # Ekstrak total per bulan
    total_pasien_per_bulan = []
    total_omset_per_bulan = []
    
    for bulan in selected_months:
        pasien_col = f"PASIEN {bulan}"
        omset_col = f"OMSET {bulan}"
        
        if pasien_col in aggregated_dict:
            total_pasien_per_bulan.append(aggregated_dict[pasien_col] or 0)
        else:
            total_pasien_per_bulan.append(0)
            
        if omset_col in aggregated_dict:
            total_omset_per_bulan.append(aggregated_dict[omset_col] or 0)
        else:
            total_omset_per_bulan.append(0)
    
    # Tombol prediksi
    if st.button("📊 Tampilkan Tren Belanja"):
        st.subheader(f"Tren Bulan {bulan_start} – {bulan_end}")
        
        # Tampilkan ringkasan
        col1, col2 = st.columns(2)
        col1.metric("Total Pasien", f"{int(sum(total_pasien_per_bulan)):,}")
        col2.metric("Total Omset", f"Rp {int(sum(total_omset_per_bulan)):,}")
        
        # Visualisasi tren
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Plot pasien
        ax1.plot(selected_months, total_pasien_per_bulan, color='tab:blue', marker='o', label='Jumlah Pasien')
        ax1.set_ylabel("Jumlah Pasien", color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Plot omset (sumbu kanan)
        ax2 = ax1.twinx()
        ax2.plot(selected_months, total_omset_per_bulan, color='tab:red', marker='s', label='Omset (Rp)')
        ax2.set_ylabel("Omset (Rupiah)", color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'Rp {int(x):,}'))
        
        # Judul & grid
        plt.title(f"Tren Pasien & Omset ({jenis})")
        ax1.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        
        # Gabungkan legenda
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        
        st.pyplot(fig)
        
        # Opsional: tampilkan tabel
        with st.expander("📋 Lihat Data Per Bulan"):
            result_data = []
            for i, bulan in enumerate(selected_months):
                result_data.append({
                    "Bulan": bulan,
                    "Jumlah Pasien": int(total_pasien_per_bulan[i]),
                    "Total Omset": int(total_omset_per_bulan[i])
                })
            
            # Buat DataFrame Pandas untuk ditampilkan
            import pandas as pd
            result_df = pd.DataFrame(result_data)
            result_df["Total Omset"] = result_df["Total Omset"].apply(lambda x: f"Rp {x:,}")
            st.dataframe(result_df)
            
            # Tampilkan juga informasi tentang data yang diproses
            st.info(f"Data diproses dari {row_count} baris data menggunakan PySpark")

if __name__ == "__main__":
    main()
