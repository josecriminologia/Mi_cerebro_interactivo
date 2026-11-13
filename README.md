# Explorador del cerebro

Recurso docente interactivo para explorar en 3D qué estructuras del cerebro participan en la emoción, la memoria, la recompensa y el control, con datos reales de neuroimagen.

**Autor:** José Luis Labiaga, doctorando. Universidad Francisco de Vitoria.

## Qué contiene

- **Anatomía:** diez estructuras del circuito límbico y de recompensa, con su localización y función.
- **Tareas:** catorce mapas de activación del Human Connectome Project (emoción, recompensa, cognición social, lenguaje, memoria de trabajo, percepción y movimiento).
- **Conceptos:** diecisiete mapas de NeuroQuery (miedo, estrés postraumático, agresión, impulsividad, craving, empatía, etc.).
- **Comparar:** actividad media de cada estructura en cada tarea.

Toda la web está en un único archivo, `index.html`, que funciona sin conexión.

## Fuentes

- Plantilla MNI152 (Fonov et al., 2011).
- Atlas de Jülich (Eickhoff et al., 2005), Harvard-Oxford (Makris et al., 2006; Desikan et al., 2006) y atlas subcortical de Pauli et al. (2018).
- Human Connectome Project (Van Essen et al., 2013; Barch et al., 2013), mapas de grupo descargados de NeuroVault (Gorgolewski et al., 2015), colección 457.
- NeuroQuery (Dockès et al., 2020).

Programado en Python (nilearn, nibabel, scikit-image y Plotly), con código generado con ayuda de inteligencia artificial y revisado por el autor.

## Condiciones de uso de los datos

Data were provided [in part] by the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems Neuroscience at Washington University.

Los mapas derivados del Human Connectome Project se redistribuyen conforme a las WU-Minn HCP Consortium Open Access Data Use Terms.
