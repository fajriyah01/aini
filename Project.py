import numpy as np 
from PIL import Image 

# Fungsi untuk mengonversi nilai RGB ke dalam format yang diinginkan
def convert_to_numeric_rgb(pixel):
    return np.array([pixel[0], pixel[1], pixel[2]])

# Fungsi untuk mengonversi nilai numerik RGB kembali ke dalam format RGB asli
def convert_to_pixel_format(pixel):
    return (int(pixel[0]), int(pixel[1]), int(pixel[2]))

# Fungsi untuk menghitung determinan modulo 256 dari matriks 3x3
def determinant_modulo_256(matrix):
    det = np.linalg.det(matrix)
    det_mod_256 = det % 256
    return det_mod_256

# Fungsi untuk mencari nilai GCD (Greatest Common Divisor)
def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

# Fungsi untuk menghasilkan matriks acak 3x3
def generate_random_matrix():
    return np.random.randint(10, 100, size=(3, 3))

# Fungsi untuk melakukan enkripsi Hill Cipher
def hill_cipher_encrypt(plaintext, matrix):
    plaintext_vector = np.array(plaintext)
    encrypted_vector = np.dot(matrix, plaintext_vector) % 256
    return encrypted_vector

# Fungsi untuk menghitung invers matriks 3x3 modulo 256
def inverse_matrix_modulo_256(matrix):
    det = np.linalg.det(matrix)
    det_inv = pow(int(det), -1, 256)  # Menghitung invers determinan modulo 256

    # Menghitung matriks adjoin
    adjoint_matrix = np.zeros_like(matrix, dtype=int)
    adjoint_matrix[0, 0] = (matrix[1, 1] * matrix[2, 2] - matrix[1, 2] * matrix[2, 1]) % 256
    adjoint_matrix[0, 1] = (matrix[0, 2] * matrix[2, 1] - matrix[0, 1] * matrix[2, 2]) % 256
    adjoint_matrix[0, 2] = (matrix[0, 1] * matrix[1, 2] - matrix[0, 2] * matrix[1, 1]) % 256
    adjoint_matrix[1, 0] = (matrix[1, 2] * matrix[2, 0] - matrix[1, 0] * matrix[2, 2]) % 256
    adjoint_matrix[1, 1] = (matrix[0, 0] * matrix[2, 2] - matrix[0, 2] * matrix[2, 0]) % 256
    adjoint_matrix[1, 2] = (matrix[0, 2] * matrix[1, 0] - matrix[0, 0] * matrix[1, 2]) % 256
    adjoint_matrix[2, 0] = (matrix[1, 0] * matrix[2, 1] - matrix[1, 1] * matrix[2, 0]) % 256
    adjoint_matrix[2, 1] = (matrix[0, 1] * matrix[2, 0] - matrix[0, 0] * matrix[2, 1]) % 256
    adjoint_matrix[2, 2] = (matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]) % 256

    # Mengalikan adjoin dengan invers determinan modulo 256
    inverse_matrix = (adjoint_matrix * det_inv) % 256

    return inverse_matrix.astype(int)

def mod_inverse(a, m):
    m0, x0, x1 = m, 0, 1
    while a > 1:
        q = a // m
        m, a = a % m, m
        x0, x1 = x1 - q * x0, x0
    return x1 + m0 if x1 < 0 else x1

def is_prime(number):
    if number < 2:
        return False
    if number <= 3:
        return True
    if number % 2 == 0 or number % 3 == 0:
        return False
    i = 5
    while i * i <= number:
        if number % i == 0 or number % (i + 2) == 0:
            return False
        i += 6
    return True

def generate_key(p, q):
    n = p * q
    phi = (p - 1) * (q - 1)
    print("Nilai Phi:", phi)
    while True:
        e = input("Masukkan nilai e, dengan 1 < e < phi dan relatif prima dengan phi: ")
        if not e.isdigit():
            print("Nilai e harus berupa bilangan.")
            continue
        e = int(e)
        if gcd(phi, e) != 1:
            print("Nilai e tidak memenuhi syarat. Harus relatif prima dengan phi.")
        else:
            break
    d = mod_inverse(e, phi)
    return (e, n), (d, n), phi

def encrypt_value(value, public_key):
    e, n = public_key
    cipher = pow(value, e, n)
    cipher_value = cipher % 256
    return cipher_value, cipher

def encrypt_rgb(pixel, public_key):
    encrypted_pixel = [encrypt_value(value, public_key) for value in pixel]
    return encrypted_pixel

def calculate_k(cipher, cipher_value):
    k = (cipher - cipher_value) // 256
    return k

try:
    # Pembangkitan Matriks Kunci Hill Cipher
    print("\n======PEMBANGKITAN KUNCI HILL CIPHER======")
    # Mencari matriks acak yang determinannya relatif prima dengan 256
    found = False
    while not found:
        matrix = generate_random_matrix()
        det_mod_256 = determinant_modulo_256(matrix)
        if gcd(det_mod_256, 256) == 1:
            found = True
            print("\nMatriks dengan determinan relatif prima dengan 256 ditemukan:")
            print(matrix)
            print("Determinan modulo 256:", det_mod_256)

    # Menghitung invers matriks modulo 256
    inverse = inverse_matrix_modulo_256(matrix)

    # Menampilkan hasil
    print("\nMatriks Awal:")
    print(matrix)
    print("\nInvers Matriks modulo 256:")
    print(inverse)

    # Proses Enkripsi Algoritma Hill Cipher
    print("\n======PROSES ENKRIPSI GAMBAR DENGAN HILL CIPHER======")
    while True:
        try:
            # Prompt pengguna untuk memasukkan nama file gambar
            file_path = input("Masukkan nama file gambar (dengan ekstensi BMP atau PNG): ")
            while not (file_path.split(".")[-1].upper() in ["BMP", "PNG"] and len(file_path.split(".")) == 2 and file_path.split(".")[0]):
                print("Tipe file gambar tidak didukung atau nama file tidak valid. Silakan masukkan file BMP atau PNG dengan nama file yang benar.")
                file_path = input("\nMasukkan nama file gambar (dengan ekstensi BMP atau PNG): ")

            # Membuka gambar
            image = Image.open(file_path)
            break
        except FileNotFoundError:
            print("File gambar tidak ditemukan. Silakan coba lagi.")

    while True:
        try:
            encrypted_file_path_hc = input("Masukkan nama file untuk hasil enkripsi Hill Cipher (dengan ekstensi BMP atau PNG): ")
            while not (encrypted_file_path_hc.split(".")[-1].upper() in ["BMP", "PNG"] and len(encrypted_file_path_hc.split(".")) == 2 and encrypted_file_path_hc.split(".")[0]):
                print("Tipe file gambar tidak didukung atau nama file tidak valid. Silakan masukkan file BMP atau PNG dengan nama file yang benar.")
                encrypted_file_path_hc = input("\nMasukkan nama file untuk hasil enkripsi Hill Cipher (dengan ekstensi BMP atau PNG): ")
            break
        except FileNotFoundError:
            print("File gambar tidak ditemukan. Silakan coba lagi.")

    # Mendapatkan ukuran gambar
    width, height = image.size

    # Membuat array numpy untuk menyimpan hasil enkripsi
    encrypted_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Mengambil nilai pixel dari setiap titik di gambar
    for y in range(height):
        for x in range(width):
            # Mendapatkan nilai RGB dari setiap pixel
            pixel = image.getpixel((x, y))

            # Enkripsi nilai RGB menggunakan Hill Cipher
            encrypted_rgb = hill_cipher_encrypt(convert_to_numeric_rgb(pixel), matrix)

            # Menyimpan nilai enkripsi ke dalam array numpy
            encrypted_image[y, x] = convert_to_pixel_format(encrypted_rgb)

    # Membuat gambar dari array numpy hasil enkripsi Hill Cipher
    encrypted_img = Image.fromarray(encrypted_image)

    # Menyimpan gambar hasil enkripsi Hill Cipher
    encrypted_img.save(encrypted_file_path_hc)

    print(f"Gambar berhasil dienkripsi. Hasil disimpan sebagai '{encrypted_file_path_hc}'.")

    # Pembangkitan Kunci untuk Algoritma RSA
    print("\n======PEMBANGKITAN KUNCI RSA======")
    while True:
        p = input("Masukkan bilangan prima p: ")
        while not p.isdigit() or not is_prime(int(p)):
            print("Bilangan yang dimasukkan bukan bilangan prima.")
            p = input("Masukkan bilangan prima p: ")
        p = int(p)

        q = input("Masukkan bilangan prima q: ")
        while not q.isdigit() or not is_prime(int(q)):
            print("Bilangan yang dimasukkan bukan bilangan prima.")
            q = input("Masukkan bilangan prima q: ")
        q = int(q)

        if p == q:
            print("p dan q harus berbeda! Silakan masukkan ulang nilai p dan q.")
        else:
            break

    public_key, private_key, phi = generate_key(p, q)

    # Menampilkan kunci publik, kunci privat, dan nilai phi
    print("\nKunci Publik:", public_key)
    print("Kunci Privat:", private_key)

    # Proses Enkripsi Gambar dengan Algoritma RSA
    print("\n======PROSES ENKRIPSI GAMBAR DENGAN ALGORITMA RSA======")
    while True:
        try:
            encrypted_file_path_rsa = input("Masukkan nama file untuk hasil enkripsi RSA (dengan ekstensi BMP atau PNG): ")
            while not (encrypted_file_path_rsa.split(".")[-1].upper() in ["BMP", "PNG"] and len(encrypted_file_path_rsa.split(".")) == 2 and encrypted_file_path_rsa.split(".")[0]):
                print("Tipe file gambar tidak didukung atau nama file tidak valid. Silakan masukkan file BMP atau PNG dengan nama file yang benar.")
                encrypted_file_path_rsa = input("\nMasukkan nama file untuk hasil enkripsi RSA (dengan ekstensi BMP atau PNG): ")

            # Membuka gambar hasil enkripsi Hill Cipher
            image = Image.open(encrypted_file_path_hc)
            break
        except FileNotFoundError:
            print("File gambar tidak ditemukan. Silakan coba lagi.")

    # Mendapatkan ukuran gambar
    width, height = image.size

    # Membuat array numpy untuk menyimpan hasil enkripsi
    encrypted_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Membuat array untuk menyimpan nilai k
    k_values = np.zeros((height * width, 3), dtype=np.int32)

    # Mengambil nilai pixel dari setiap titik di gambar
    index = 0
    for y in range(height):
        for x in range(width):
            # Mendapatkan nilai RGB dari setiap pixel
            pixel = image.getpixel((x, y))

            # Enkripsi nilai RGB
            encrypted_pixel = encrypt_rgb(pixel, public_key)

            for i, (cipher_value, cipher) in enumerate(encrypted_pixel):
                # Menyimpan nilai enkripsi ke dalam array numpy
                encrypted_image[y, x, i] = cipher_value

                # Hitung nilai k
                k = calculate_k(cipher, cipher_value)

                # Simpan nilai k ke dalam array
                k_values[index, i] = k

            index += 1

    # Membuat gambar dari array numpy hasil enkripsi
    encrypted_img = Image.fromarray(encrypted_image)

    # Menyimpan gambar hasil enkripsi
    encrypted_img.save(encrypted_file_path_rsa)

    # Simpan nilai k sebagai file teks
    k_file_path = encrypted_file_path_rsa.split('.')[0] + "_k.txt"
    np.savetxt(k_file_path, k_values, fmt='%d')

    print(f"Gambar berhasil dienkripsi. Hasil disimpan sebagai '{encrypted_file_path_rsa}'.")
    print(f"Nilai k berhasil disimpan sebagai '{k_file_path}'.")

except ValueError as ve:
    print(f"Error: {ve}")
except Exception as e:
    print(f"Terjadi kesalahan: {str(e)}")
