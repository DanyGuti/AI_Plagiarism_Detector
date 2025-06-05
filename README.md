# AI_Plagiarism_Detector

## Derechos reservados Tec de Monterrey ® Campus Querétaro

## Proyecto realizado por:

<table style="margin:auto;">
  <tr>
    <th style="text-align:center;">Estudiante</th>
    <th style="text-align:center;">Matrícula</th>
    <th style="text-align:center;">Correo</th>
  </tr>
  <tr>
    <td style="text-align:center;">Daniel Gutiérrez Gómez</td>
    <td style="text-align:center;">A01068056</td>
    <td style="text-align:center;">danyguti2001@hotmail.com</td>
  </tr>
  <tr>
    <td style="text-align:center;">Olimpia Helena García Huerta</td>
    <td style="text-align:center;">A01708462</td>
    <td style="text-align:center;">A01708462@tec.mx</td>
  </tr>
  <tr>
    <td style="text-align:center;">Diego Ernesto Sandoval Vargas</td>
    <td style="text-align:center;">A01709113</td>
    <td style="text-align:center;">A01709113@tec.mx</td>
  </tr>
</table>

## Profesores de Bloque Integrador TC3002B
<table style="margin:auto;">
  <tr>
    <th style="text-align:center;">Profesor</th>
    <th style="text-align:center;">Módulo</th>
  </tr>
  <tr>
    <td style="text-align:center;">Pedro Oscar Pérez Murueta</td>
    <td style="text-align:center;">Metodología de Investigación | Compiladores</td>
  </tr>
  <tr>
    <td style="text-align:center;">Benjamin Valdés Aguirre</td>
    <td style="text-align:center;">Inteligencia Artificial</td>
  </tr>
  <tr>
    <td style="text-align:center;">Manuel Iván Casillas del Llano</td>
    <td style="text-align:center;">Métodos Cuantitativos</td>
  </tr>
</table>

---

## Descripción

<p style="text-align:justify;">
    El presente proyecto, trata de la <b>detección de plagio de códigos Java</b> con base a distintas taxonomías denominadas por artículos de investigación con el uso de un modelo de inteligencia artificial y su alimentación, dos datasets. Enfoque de detección de plagio en un nivel estructural.
</p>

<p style="text-align:justify;">
    Integración de módulos en el proyecto, con el uso de <b>algoritmos de exploración de hiper-parámetros configuracionales del modelo</b> (métodos cuantitativos), el uso de herramienta
    para la generación de los datos pertinentes para su análisis (ANTLR 4) generando árboles de sintaxis mejor conocidos como <b>ASTs</b>, herramienta que utiliza un tipo de <b>parser ALL(*)</b> (módulo de compiladores) y un modelo de inteligencia artificial que utiliza dos capas, capa de extracción de características con el uso de capas tipo <b>"Embedding" y una capa de aprendizaje profundo CNN</b> con función sigmoide para el reconocimiento de plagio de manera clasificatoria binaria.
</p>

## Proceso de construcción experimental del modelo
<ol style="font-size:14px; font-weight:550;">
    <li>
        Investigación de datasets (códigos fuente o snippets).
    </li>
    <li>
        Verificación de datos recolectados.
    </li>
    <li>
        Toma de decisión sobre el formato en datos necesario para el modelo de Inteligencia Artificial
    </li>
    <li>
        Pre-procesamiento de datos.
    </li>
    <li>
        Separación de datos: Entrenamiento, validación y pruebas (60%, 20% y 20%).
    </li>
    <li>
        Generación del modelo de Inteligencia Artificial.
    </li>
    <li>
        Configuración de hiper-parámetros.
    </li>
        Análisis de resultados.
    <li>
         Iteración del paso 8 y 9 hasta llegar a un grado de generalización.
    </li>
</ol>


## ¿Cómo generar el modelo de Inteligencia Artificial?
### Requerimientos:
* Computadora con procesador GPU
* Dataset `IR-Plag dataset`, por medio de búsqueda en algún repositorio.
* Dataset descargado de manera local desde [Zenodo](https://zenodo.org/records/7332790).

### Pasos
1. Descarga de requerimientos, posicionamiento en directorio root del proyecto, ejecutar desde la terminal o línea de comandos: `pip install -r requirements.txt `
2. Posicionar datasets en directorio raíz del proyecto
3. Pipeline completo de modelo de Inteligencia Artificial: Ir al file: `/src/main.py`, descomentar las funciones "import" del archivo "train", descomentar las siguientes líneas de código en el siguiente rango [31,56]
4. Ejecutar módulo main desde la terminal o línea de comandos: `python3 main.py`

