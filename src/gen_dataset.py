import cv2
import numpy as np
import os
import argparse

def generate_synthetic_image(size, noise_percent, num_lines, num_circles, output_dir):
    """
    Genera un'immagine sintetica per il benchmarking della Hough Transform.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Creazione di un'immagine nera vuota (sfondo zero)
    img = np.zeros((size, size), dtype=np.uint8)

    # 2. Aggiunta delle linee (spessore 1 pixel)
    for _ in range(num_lines):
        x1, y1 = np.random.randint(0, size, 2)
        x2, y2 = np.random.randint(0, size, 2)
        cv2.line(img, (x1, y1), (x2, y2), 255, 1)

    # 3. Aggiunta dei cerchi (spessore 1 pixel)
    for _ in range(num_circles):
        center_x, center_y = np.random.randint(0, size, 2)
        # Raggio limitato per evitare che il cerchio sia enorme e fuori dall'immagine
        max_radius = min(size // 4, 500) 
        radius = np.random.randint(10, max(11, max_radius))
        cv2.circle(img, (center_x, center_y), radius, 255, 1)

    # 4. Aggiunta del rumore (pixel "accesi" casualmente per il load balancing)
    if noise_percent > 0:
        # Calcoliamo quanti pixel accendere in base alla percentuale
        num_noise_pixels = int((size * size) * (noise_percent / 100.0))
        
        # Generiamo coordinate casuali
        noise_x = np.random.randint(0, size, num_noise_pixels)
        noise_y = np.random.randint(0, size, num_noise_pixels)
        
        # Accendiamo i pixel
        img[noise_y, noise_x] = 255

    # 5. Salvataggio dell'immagine
    filename = f"synth_{size}x{size}_L{num_lines}_C{num_circles}_N{noise_percent}pct.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, img)
    print(f"Salvato: {filepath} | Pixel totali: {size*size} | Rumore: {noise_percent}%")

if __name__ == "__main__":
    # Configurazione tramite riga di comando per automatizzare sul cluster
    parser = argparse.ArgumentParser(description="Generatore Dataset HT per HPC")
    parser.add_argument("--size", type=int, default=1024, help="Dimensione N dell'immagine NxN")
    parser.add_argument("--noise", type=float, default=1.0, help="Percentuale di rumore (es. 5 per 5%)")
    parser.add_argument("--lines", type=int, default=5, help="Numero di linee da generare")
    parser.add_argument("--circles", type=int, default=0, help="Numero di cerchi da generare")
    parser.add_argument("--outdir", type=str, default="hpc_dataset", help="Cartella di output")
    
    args = parser.parse_args()
    
    generate_synthetic_image(args.size, args.noise, args.lines, args.circles, args.outdir)
