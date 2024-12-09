from flask import Flask, request, jsonify, render_template
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Inisialisasi aplikasi Flask
def get_weather_data(lat, lon):
    # Visual Crossing Weather API endpoint
    api_key = "LMSDGRRGVNZ8NLHPRG6NJ7DAA"  # Ganti dengan API key Anda yang valid
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}?unitGroup=us&key={api_key}&contentType=json"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            temperatures = [day['temp'] for day in data['days']]
            humidity = [day['humidity'] for day in data['days']]
            
            # Konversi suhu rata-rata dari Fahrenheit ke Celsius
            avg_temp_c = round((sum(temperatures) / len(temperatures) - 32) * 5/9, 3)
            avg_hum_c = round(sum(humidity) / len(humidity))
            
            return {
                'rata_suhu_c': avg_temp_c,
                'kelembaban': avg_hum_c,
            }
        else:
            print("Error: Tidak dapat mengambil data cuaca.")
            return None
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None

# Membaca dataset
file_path = 'Dataset.csv'  
dataset = pd.read_csv(file_path)

# Memisahkan fitur dan target
X = dataset[['Temperature', 'Humidity', 'Jenis_Tanah']]
y = dataset['Crop']

# Mengonversi data kategorikal menjadi numerik menggunakan one-hot encoding
X = pd.get_dummies(X, columns=['Jenis_Tanah'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inisialisasi model Random Forest dengan parameter yang dioptimalkan
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    max_features='log2',
    min_samples_leaf=4,
    min_samples_split=4,
    criterion='entropy',
    n_jobs=-1,
    random_state=42
)

# Melatih model
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi model: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Fungsi untuk memproses input petani dengan klasifikasi awal berdasarkan jenis tanah
def rekomendasi_tanaman(model, jenis_tanah, suhu, kelembapan):
    # Mendapatkan kolom jenis tanah dari dataset
    jenis_tanah_columns = [col for col in X.columns if "Jenis_Tanah" in col]

    # Memastikan input jenis tanah valid
    valid_tanah = False
    jenis_tanah_filter = None
    for col in jenis_tanah_columns:
        if jenis_tanah.lower() in col.lower():
            jenis_tanah_filter = col
            valid_tanah = True
            break

    # Jika jenis tanah tidak valid, hentikan proses
    if not valid_tanah:
        print(f"Jenis tanah '{jenis_tanah}' tidak ditemukan atau tidak valid.")
        return None

    # Filter dataset berdasarkan jenis tanah
    filtered_indices = X_test[X_test[jenis_tanah_filter] == 1].index
    X_test_filtered = X_test.loc[filtered_indices]
    y_test_filtered = y_test.loc[filtered_indices]

    # Jika tidak ada data dengan jenis tanah yang cocok
    if len(X_test_filtered) == 0:
        print(f"Tidak ada tanaman yang cocok untuk jenis tanah '{jenis_tanah}'.")
        return None

    # Menyiapkan input petani dalam format fitur model
    input_data = {col: 0 for col in X.columns}
    input_data["Temperature"] = suhu
    input_data["Humidity"] = kelembapan
    input_data[jenis_tanah_filter] = 1  # Hanya jenis tanah yang relevan diaktifkan

    input_df = pd.DataFrame([input_data])

    # Prediksi tanaman
    prediksi = model.predict(input_df)
    return prediksi[0], X_test_filtered, y_test_filtered

# Fungsi untuk memproses input petani dengan memfilter jenis tanah terlebih dahulu
def rekomendasi_tanaman_berdasarkan_jenis_tanah(model, jenis_tanah, suhu, kelembapan, X_train, y_train):
    # Filter data berdasarkan jenis tanah
    jenis_tanah_columns = [col for col in X_train.columns if "Jenis_Tanah" in col]
    jenis_tanah_filter = None

    for col in jenis_tanah_columns:
        if jenis_tanah.lower() in col.lower():
            jenis_tanah_filter = col
            break

    if not jenis_tanah_filter:
        print(f"Jenis tanah '{jenis_tanah}' tidak ditemukan dalam data.")
        return None

    # Filter data pelatihan hanya untuk jenis tanah yang sesuai
    filtered_indices = X_train[X_train[jenis_tanah_filter] == 1].index
    X_train_filtered = X_train.loc[filtered_indices]
    y_train_filtered = y_train.loc[filtered_indices]

    if len(X_train_filtered) == 0:
        print(f"Tidak ada data pelatihan yang sesuai untuk jenis tanah '{jenis_tanah}'.")
        return None

    # Buat DataFrame untuk input petani
    input_data = {col: 0 for col in X_train.columns}
    input_data["Temperature"] = suhu
    input_data["Humidity"] = kelembapan
    input_data[jenis_tanah_filter] = 1
    input_df = pd.DataFrame([input_data])

    # Latih ulang model pada data yang difilter
    model.fit(X_train_filtered, y_train_filtered)

    # Prediksi tanaman
    prediksi = model.predict(input_df)
    return prediksi[0]
def get_alternative_crops(X_train, y_train, jenis_tanah):
    """
    Mencari daftar tanaman alternatif untuk jenis tanah tertentu dengan detail rentang suhu dan kelembapan
    """
    # Filter kolom jenis tanah
    jenis_tanah_columns = [col for col in X_train.columns if "Jenis_Tanah" in col]
    jenis_tanah_filter = None

    for col in jenis_tanah_columns:
        if jenis_tanah.lower() in col.lower():
            jenis_tanah_filter = col
            break

    if not jenis_tanah_filter:
        print(f"Jenis tanah '{jenis_tanah}' tidak ditemukan.")
        return None

    # Filter data pelatihan untuk jenis tanah yang sesuai
    filtered_indices = X_train[X_train[jenis_tanah_filter] == 1].index
    X_train_filtered = X_train.loc[filtered_indices]
    y_train_filtered = y_train.loc[filtered_indices]

    if len(X_train_filtered) == 0:
        print(f"Tidak ada data pelatihan untuk jenis tanah '{jenis_tanah}'.")
        return None

    # Buat dictionary untuk menyimpan detail tanaman
    tanaman_details = {}
    for tanaman in y_train_filtered.unique():
        tanaman_data = X_train_filtered[y_train_filtered == tanaman]
        tanaman_details[tanaman] = {
            'suhu_min': tanaman_data['Temperature'].min(),
            'suhu_max': tanaman_data['Temperature'].max(),
            'kelembapan_min': tanaman_data['Humidity'].min(),
            'kelembapan_max': tanaman_data['Humidity'].max()
        }

    return tanaman_details

def hitung_akurasi_keseluruhan_dengan_alternatif(
    model, 
    X_test, 
    y_test, 
    X_train, 
    y_train, 
    jenis_tanah, 
    suhu, 
    kelembapan, 
    toleransi=5
):
    """
    Menghitung akurasi dengan mencari tanaman alternatif jika tidak ada tanaman yang cocok
    """
    jenis_tanah_columns = [col for col in X_test.columns if "Jenis_Tanah" in col]
    jenis_tanah_filter = None
    for col in jenis_tanah_columns:
        if jenis_tanah.lower() in col.lower():
            jenis_tanah_filter = col
            break

    if not jenis_tanah_filter:
        print(f"Jenis tanah '{jenis_tanah}' tidak ditemukan di data.")
        return None, None

    X_test_filtered = X_test[X_test[jenis_tanah_filter] == 1]
    y_test_filtered = y_test[X_test_filtered.index]

    if len(X_test_filtered) == 0:
        print(f"Tidak ada data untuk jenis tanah '{jenis_tanah}'.")
        return None, None

    X_test_final = X_test_filtered[
        (X_test_filtered["Temperature"].between(suhu - toleransi, suhu + toleransi)) &
        (X_test_filtered["Humidity"].between(kelembapan - toleransi, kelembapan + toleransi))
    ]
    y_test_final = y_test_filtered[X_test_final.index]

    # Jika tidak ada tanaman yang cocok, cari alternatif
    if len(X_test_final) == 0:
        tanaman_details = get_alternative_crops(X_train, y_train, jenis_tanah)
        return None, tanaman_details

    y_pred_final = model.predict(X_test_final)
    akurasi_keseluruhan = accuracy_score(y_test_final, y_pred_final) * 100
    return akurasi_keseluruhan, None

# Modifikasi bagian utama untuk menggunakan fungsi baru
def main():
    # [Kode sebelumnya untuk memuat data dan melatih model]
    
    # Contoh input dari petani
    print("\n=== Input Petani ===")
    lat = input("Masukkan latitude: ")
    lon = input("Masukkan longitude: ")

    # Mendapatkan data cuaca
    weather_data = get_weather_data(lat, lon)

    if not weather_data:
        print("Gagal mendapatkan data cuaca. Mohon cek koneksi internet atau koordinat.")
        exit()

    suhu = weather_data['rata_suhu_c']
    kelembapan = weather_data['kelembaban']

    print("\nData Cuaca:")
    print(f"Rata-rata Suhu (°C): {suhu}")
    print(f"Kelembaban (%): {kelembapan}")

    jenis_tanah = input("Masukkan jenis tanah: ")

    # Menampilkan rekomendasi tanaman
    akurasi, tanaman_details = hitung_akurasi_keseluruhan_dengan_alternatif(
        model, X_test, y_test, X_train, y_train, jenis_tanah, suhu, kelembapan
    )

    if akurasi is not None:
        rekomendasi = rekomendasi_tanaman_berdasarkan_jenis_tanah(
            model, jenis_tanah, suhu, kelembapan, X_train, y_train
        )
        if rekomendasi:
            print("\nRekomendasi tanaman berdasarkan input Anda:", rekomendasi)
    else:
        # Jika tidak ada tanaman yang cocok
        if tanaman_details:
            print(f"\nTidak ada tanaman yang cocok dengan {suhu}°C dan kelembapan {kelembapan}% untuk jenis tanah '{jenis_tanah}'.")
            print("\n=== Rekomendasi Detail Tanaman ===")
            for tanaman, detail in tanaman_details.items():
                print(f"Tanaman: {tanaman}")
                print(f"Suhu: {detail['suhu_min']:.2f}°C - {detail['suhu_max']:.2f}°C")
                print(f"Kelembapan: {detail['kelembapan_min']:.2f}% - {detail['kelembapan_max']:.2f}%")
                print()
        else:
            print("Tidak dapat menemukan rekomendasi tanaman.")
app = Flask(__name__)

# Rute untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Rute untuk input petani dan rekomendasi tanaman
@app.route('/rekomendasi', methods=['POST'])
def rekomendasi():
    try:
        lat = request.form.get('latitude')
        lon = request.form.get('longitude')
        jenis_tanah = request.form.get('jenis_tanah')

        # Mendapatkan data cuaca
        weather_data = get_weather_data(lat, lon)
        if not weather_data:
            return jsonify({"error": "Gagal mendapatkan data cuaca. Mohon cek koneksi internet atau koordinat."}), 500

        suhu = weather_data['rata_suhu_c']
        kelembapan = weather_data['kelembaban']

        # Menampilkan rekomendasi tanaman
        akurasi, tanaman_details = hitung_akurasi_keseluruhan_dengan_alternatif(
            model, X_test, y_test, X_train, y_train, jenis_tanah, suhu, kelembapan
        )

        if akurasi is not None:
            rekomendasi = rekomendasi_tanaman_berdasarkan_jenis_tanah(
                model, jenis_tanah, suhu, kelembapan, X_train, y_train
            )
            if rekomendasi:
                return jsonify({
                    "rata_suhu_c": suhu,
                    "kelembaban": kelembapan,
                    "jenis_tanah": jenis_tanah,
                    "rekomendasi_tanaman": rekomendasi
                })
        else:
            if tanaman_details:
                return jsonify({
                    "rata_suhu_c": suhu,
                    "kelembaban": kelembapan,
                    "jenis_tanah": jenis_tanah,
                    "alternatif_tanaman": tanaman_details
                })
            else:
                return jsonify({"error": "Tidak dapat menemukan rekomendasi tanaman."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Menjalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
