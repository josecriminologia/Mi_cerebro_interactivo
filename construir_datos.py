# ============================================================
#  Genera los datos de la web "Explorador del cerebro":
#  mallas 3D de anatomía, de las tareas (Human Connectome Project)
#  y de los conceptos (NeuroQuery), más la tabla estructura x tarea.
#  Escribe datos/anatomia.js, datos/tareas.js y datos/conceptos.js.
#  Los textos NO están aquí: están en textos.js.
# ============================================================
import base64, json, sys, warnings
from pathlib import Path
import numpy as np
from scipy import ndimage
from skimage import measure
from nilearn import datasets, image
warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
DATOS = AQUI / "datos"; DATOS.mkdir(exist_ok=True)
HCP = Path.home() / "nilearn_data" / "neurovault" / "collection_457"
NEUROQUERY = Path(sys.argv[1]) if len(sys.argv) > 1 else None   # carpeta neuroquery_model

# ---------- utilidades ----------
def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

def codifica(v, c):
    """Vértices en centésimas de mm (int16) y caras (uint16 o uint32), en base64."""
    ancho = 16 if len(v) < 65536 else 32
    return dict(n=int(len(v)), v=b64(np.round(v * 100).astype("<i2")),
                f=b64(c.astype("<u2" if ancho == 16 else "<u4")), w=ancho)

def malla(vol, afin, nivel=0.5, paso=1):
    if vol.max() <= nivel:
        return None
    v, c, _, _ = measure.marching_cubes(vol, level=nivel, step_size=paso)
    v = (np.c_[v, np.ones(len(v))] @ afin.T)[:, :3]
    return codifica(v, c)

def limpia(m, minimo=60):
    et, n = ndimage.label(m)
    if n == 0:
        return m
    tam = ndimage.sum(m, et, range(1, n + 1))
    return np.isin(et, 1 + np.flatnonzero(tam >= minimo))

def mancha(mascara_t, afin, paso=2):
    return malla(ndimage.gaussian_filter(limpia(mascara_t).astype(np.float32), 0.7), afin, paso=paso)

def escribe(nombre, variable, objeto):
    destino = DATOS / nombre
    destino.write_text(f"window.{variable} = " + json.dumps(objeto, ensure_ascii=False, separators=(",", ":")) + ";\n",
                       encoding="utf-8")
    print(f"   {nombre}: {destino.stat().st_size / 1e6:.1f} MB")

# ---------- 1. Anatomía ----------
jul  = datasets.fetch_atlas_juelich("maxprob-thr25-1mm")
hsub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")
hcor = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")
pau  = datasets.fetch_atlas_pauli_2017(atlas_type="deterministic")
J, HS, HC, P = (image.load_img(a.maps) for a in (jul, hsub, hcor, pau))
def donde(atlas, texto):
    return [i for i, n in enumerate(atlas.labels) if texto in str(n)]
def varias(atlas, textos):
    return sorted({i for t in textos for i in donde(atlas, t)})

# Lóbulos: cada uno agrupa regiones del atlas cortical Harvard-Oxford (por el nombre de la etiqueta).
# La ínsula queda fuera porque ya tiene ficha propia.
LOBULOS = {
    "frontal":   ["Frontal Pole", "Superior Frontal", "Middle Frontal", "Inferior Frontal", "Precentral",
                  "Frontal Medial", "Juxtapositional", "Subcallosal", "Paracingulate", "Cingulate Gyrus, anterior",
                  "Frontal Orbital", "Frontal Operculum", "Central Opercular"],
    "parietal":  ["Postcentral", "Superior Parietal", "Supramarginal", "Angular", "Cingulate Gyrus, posterior",
                  "Precuneous", "Parietal Operculum"],
    "temporal":  ["Temporal Pole", "Superior Temporal", "Middle Temporal", "Inferior Temporal", "Parahippocampal",
                  "Temporal Fusiform", "Temporal Occipital Fusiform", "Planum Polare", "Heschl", "Planum Temporale"],
    "occipital": ["Lateral Occipital", "Intracalcarine", "Cuneal", "Lingual", "Occipital Fusiform",
                  "Supracalcarine", "Occipital Pole"],
}

def mascara_cerebelo():
    """El atlas Harvard-Oxford no incluye el cerebelo. Se delimita por exclusión: lo que queda dentro
    del cerebro, en la fosa posterior, sin asignar a ninguna región cortical, subcortical ni al tronco."""
    hs0 = image.load_img(datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr0-1mm").maps)
    hc0 = image.load_img(datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-1mm").maps)
    m1 = image.resample_to_img(datasets.load_mni152_brain_mask(resolution=1), hs0, interpolation="nearest",
                               force_resample=True, copy_header=True)
    dentro = ndimage.binary_erosion(m1.get_fdata() > 0, iterations=2)
    asignado = (hs0.get_fdata() > 0) | (hc0.get_fdata() > 0)
    ijk = np.indices(dentro.shape).reshape(3, -1).T
    mm = (np.c_[ijk, np.ones(len(ijk))] @ hs0.affine.T)[:, :3].reshape(dentro.shape + (3,))
    fosa = (mm[..., 1] < -32) & (mm[..., 2] < -6)
    cand = ndimage.binary_opening(dentro & ~asignado & fosa, iterations=3)
    et, n = ndimage.label(cand)
    mayor = 1 + int(np.argmax(ndimage.sum(cand, et, range(1, n + 1))))
    return ndimage.binary_fill_holes(ndimage.binary_closing(et == mayor, iterations=3))

# id, atlas, etiquetas, color, ¿entra en la tabla de comparación?
ESTRUCTURAS = [
    ("hipocampo",   J,  [9, 10, 13],                    "#1E9E8A", True),
    ("entorrinal",  J,  [11],                           "#7D5BA6", True),
    ("amigdala",    J,  [1, 2, 3],                      "#E3A008", True),
    ("accumbens",   HS, donde(hsub, "Accumbens"),       "#C0399F", True),
    ("atv",         P,  donde(pau, "VTA"),              "#0E7490", False),
    ("hipotalamo",  P,  donde(pau, "HTH"),              "#D9622B", True),
    ("pfmedial",    HC, donde(hcor, "Frontal Medial"),  "#4C8C2B", True),
    ("insula",      HC, donde(hcor, "Insular"),         "#8E6E53", True),
    ("fornix",      J,  [57],                           "#3B5BA9", False),
    ("mamilares",   J,  [25],                           "#B8860B", False),
    # Ganglios basales y tálamo
    ("talamo",      HS, donde(hsub, "Thalamus"),        "#2F9BD6", True),
    ("caudado",     HS, donde(hsub, "Caudate"),         "#E07A9B", True),
    ("putamen",     HS, donde(hsub, "Putamen"),         "#6A4FCF", True),
    ("palido",      HS, donde(hsub, "Pallidum"),        "#9BC53D", True),
    # Corteza de control. La dorsolateral se aproxima con el giro frontal medio.
    ("cingulo-anterior", HC, donde(hcor, "Cingulate Gyrus, anterior"), "#F2C14E", True),
    ("orbitofrontal",    HC, donde(hcor, "Frontal Orbital"),           "#5DBB9A", True),
    ("dorsolateral",     HC, donde(hcor, "Middle Frontal"),            "#B5651D", True),
    # Capas: envuelven a las anteriores. La web las pinta translúcidas.
    ("frontal",     HC, varias(hcor, LOBULOS["frontal"]),   "#D98C8C", True),
    ("parietal",    HC, varias(hcor, LOBULOS["parietal"]),  "#8FB8DE", True),
    ("temporal",    HC, varias(hcor, LOBULOS["temporal"]),  "#A9C77B", True),
    ("occipital",   HC, varias(hcor, LOBULOS["occipital"]), "#C9A6DD", True),
    ("cerebelo",    HS, mascara_cerebelo(),             "#E0C27A", True),
    ("tronco",      HS, donde(hsub, "Brain-Stem"),      "#9AA3AB", True),
]
CAPAS = {"frontal", "parietal", "temporal", "occipital", "cerebelo", "tronco"}
VELOS = {"cingulo-anterior", "orbitofrontal", "dorsolateral"}   # corteza que tapa lo profundo: semitransparente en la vista general
masc = datasets.load_mni152_brain_mask(resolution=2)
fondo = malla(ndimage.gaussian_filter(masc.get_fdata().astype(np.float32), 1.2), masc.affine)

MASCARAS, anat = {}, []
for ident, img, et, color, comparar in ESTRUCTURAS:
    m = et if isinstance(et, np.ndarray) else np.isin(img.get_fdata(), et)
    MASCARAS[ident] = (m, img.affine)
    grande = m.sum() > 6000
    if ident in CAPAS:      # regiones muy grandes: malla más gruesa para que el archivo no crezca demasiado
        mm = malla(ndimage.gaussian_filter(m.astype(np.float32), 1.6), img.affine, paso=3)
        anat.append(dict(id=ident, color=color, capa=True, malla=mm))
    else:
        mm = malla(ndimage.gaussian_filter(m.astype(np.float32), 1.0 if grande else 0.8), img.affine, paso=2 if grande else 1)
        anat.append(dict(id=ident, color=color, malla=mm, **({"velo": True} if ident in VELOS else {})))
    print(f"   {ident:16s} {int(m.sum()):6d} vóxeles, {mm['n']} vértices")

# ---------- 2. Tareas ----------
# id, mapa, signo, umbral a mano (None = el 5 % más activo, acotado entre 5 y 11)
# Los mapas "frente a reposo" tienen valores z muy altos y necesitan un umbral a mano más exigente.
# La tarea de lenguaje no tiene bloques de reposo: de ella solo sirve la comparación historia - cálculo.
TAREAS = [("caras-miedo", 3128, 1, None), ("ganar-perder", 3136, 1, 7.0),
          ("ganar-reposo", 3137, 1, 16.0), ("perder-reposo", 3135, 1, 16.0), ("intenciones", 3180, 1, None),
          ("historia", 3142, 1, None), ("memoria-trabajo", 3190, 1, None), ("relacional", 3172, -1, None),
          ("ver-caras", 3202, 1, None), ("ver-lugares", 3204, 1, None), ("ver-cuerpos", 3200, 1, None),
          ("ver-herramientas", 3206, 1, None), ("mano-derecha", 3161, 1, 9.0), ("mano-izquierda", 3157, 1, 9.0),
          ("pie-derecho", 3159, 1, 9.0), ("pie-izquierdo", 3155, 1, 9.0), ("lengua", 3163, 1, 9.0),
          ("movimiento-conjunto", 3152, 1, 13.0)]

def media_en_mascara(m, afin_m, d, afin_d):
    ijk = np.argwhere(m)
    mm = np.c_[ijk, np.ones(len(ijk))] @ afin_m.T
    vox = np.rint(mm @ np.linalg.inv(afin_d).T)[:, :3].astype(int)
    ok = np.all((vox >= 0) & (vox < d.shape[:3]), axis=1)
    vox = np.unique(vox[ok], axis=0)
    return float(d[vox[:, 0], vox[:, 1], vox[:, 2]].mean())

FILAS = [e[0] for e in ESTRUCTURAS if e[4]]
tareas, Z = [], []
for ident, mapa, signo, a_mano in TAREAS:
    img = image.load_img(str(HCP / f"image_{mapa}.nii.gz"))
    d = signo * np.nan_to_num(img.get_fdata()).astype(np.float32)
    cer = d != 0
    alto = a_mano or float(np.clip(np.percentile(d[cer], 95), 5, 11))
    bajo = float(np.clip(np.percentile(d[cer], 5), -11, -5))
    tareas.append(dict(id=ident, umbral=round(alto, 1),
                       mas=mancha(d >= alto, img.affine), menos=mancha(d <= bajo, img.affine, paso=2)))
    Z.append([round(media_en_mascara(*MASCARAS[f], d, img.affine), 1) for f in FILAS])
    print(f"   {ident:17s} z ≥ {alto:4.1f}")

escribe("anatomia.js", "DATOS_ANATOMIA", dict(fondo=fondo, estructuras=anat))
escribe("tareas.js", "DATOS_TAREAS", dict(tareas=tareas, filas=FILAS, columnas=[t[0] for t in TAREAS],
                                          z=np.array(Z).T.tolist()))

# ---------- 3. Conceptos (NeuroQuery) ----------
if NEUROQUERY:
    from neuroquery import NeuroQueryModel
    modelo = NeuroQueryModel.from_data_dir(str(NEUROQUERY))
    CONCEPTOS = [("miedo", "fear"), ("estres", "stress"), ("tept", "ptsd"), ("dolor", "pain"),
                 ("condicionamiento-miedo", "fear conditioning"), ("extincion", "extinction"),
                 ("agresion", "aggression"), ("impulsividad", "impulsivity"),
                 ("inhibicion", "response inhibition"), ("antisocial", "antisocial"),
                 ("empatia", "empathy"), ("juicio-moral", "moral"),
                 ("craving", "craving"), ("adiccion", "addiction"), ("interocepcion", "interoception"),
                 ("memoria-episodica", "episodic memory"), ("reconocer-caras", "face recognition")]
    plantilla = datasets.load_mni152_template(resolution=2)
    conceptos = []
    for ident, termino in CONCEPTOS:
        z4 = modelo(termino)["brain_map"]
        d4 = z4.get_fdata()
        umbral = float(max(np.percentile(d4[d4 != 0], 97), 3.5))        # el 3 % más alto del mapa
        z2 = image.resample_to_img(z4, plantilla, interpolation="continuous",
                                   force_resample=True, copy_header=True)
        conceptos.append(dict(id=ident, termino=termino, umbral=round(umbral, 1),
                              mas=mancha(z2.get_fdata() >= umbral, z2.affine)))
        print(f"   {termino:20s} z ≥ {umbral:4.1f}")
    escribe("conceptos.js", "DATOS_CONCEPTOS", dict(conceptos=conceptos))
