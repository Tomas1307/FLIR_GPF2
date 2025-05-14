import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

def marcar_imagenes_interactivamente(img_root: Path, lbl_root: Path, salida_txt="imagenes_sin_valor.txt"):
    """
    Muestra imágenes de fondo (sin anotaciones) y permite al usuario marcarlas con una tecla:
    - Presiona '0' si la imagen NO aporta información (se guarda en el txt).
    - Presiona cualquier otra tecla para pasar a la siguiente.
    """

    rutas_guardar = []
    imagenes_background = []

    for split in ["train", "val"]:
        img_dir = img_root / split
        lbl_dir = lbl_root / split

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.*")):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists() and lbl_path.read_text().strip():
                continue  # tiene anotaciones

            imagenes_background.append(img_path)

    print("Instrucciones:")
    print("- Presiona la tecla '0' si la imagen NO aporta información (se guarda).")
    print("- Presiona cualquier otra tecla para avanzar a la siguiente imagen.")
    print("- Cierra la imagen si deseas cancelar el proceso.\n")

    idx = [0]  # mutable en cierre

    def on_key(event):
        if event.key == '0':
            rutas_guardar.append(str(imagenes_background[idx[0]]))
            print(f"[{idx[0]+1}] Marcada como SIN VALOR: {imagenes_background[idx[0]].name}")
        else:
            print(f"[{idx[0]+1}] Saltada: {imagenes_background[idx[0]].name}")

        idx[0] += 1
        plt.close()

    while idx[0] < len(imagenes_background):
        img_path = imagenes_background[idx[0]]
        try:
            img = Image.open(img_path).convert("RGB")
            plt.imshow(img)
            plt.title(f"{img_path.name} ({idx[0]+1}/{len(imagenes_background)})")
            plt.axis('off')
            fig = plt.gcf()
            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.show()
        except Exception as e:
            print(f"Error al abrir imagen {img_path}: {e}")
            idx[0] += 1
            continue

    if rutas_guardar:
        with open(salida_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(rutas_guardar))
        print(f"\nSe guardaron {len(rutas_guardar)} rutas en '{salida_txt}'")
    else:
        print("\nNo se marcó ninguna imagen como sin valor.")


marcar_imagenes_interactivamente(
    img_root=Path("data/Imagenes"),
    lbl_root=Path("data/Etiquetas"),
    salida_txt="imagenes_sin_valor.txt"
)
