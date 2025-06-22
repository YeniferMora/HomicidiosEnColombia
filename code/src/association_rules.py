import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

def create_association_rules():

    # Cargar los datos
    file_path = '../data/processed/homicidios_procesado.csv'

    df = pd.read_csv(file_path)

    print(df.shape)
    df.head()



    # Seleccionamos un subconjunto de columnas categóricas que son más relevantes para encontrar patrones.
    # Excluimos identificadores, datos de ubicación muy específicos (códigos DANE) y campos de texto libre.
    features_for_mining = [
        'Grupo de edad de la victima',
        'Sexo de la victima',
        'Mes del hecho',
        'Dia del hecho',
        'Rango de Hora del Hecho X 3 Horas',
        'Escenario del Hecho',
        'Zona del Hecho',
        'Mecanismo Causal',
        'Ancestro Racial'
    ]

    df_selected = df[features_for_mining]

    # Eliminar cualquier fila con valores faltantes en nuestras columnas seleccionadas por simplicidad -----PROBABLEMENTE SOBRE-----
    df_selected.dropna(inplace=True)

    # Mostrar la forma de nuestros datos seleccionados
    print(f"Forma de los datos seleccionados después de eliminar valores faltantes: {df_selected.shape}")
    print("\nPrimeras 5 filas de los datos seleccionados:")
    print(df_selected.head())



    # Convertimos los datos categóricos a un formato one-hot encoded.
    # Esto crea una matriz binaria donde cada columna es un "ítem" y cada fila es una "transacción".
    df_encoded = pd.get_dummies(df_selected)

    # Los algoritmos en mlxtend funcionan con valores booleanos (True/False)
    df_encoded = df_encoded.astype(bool)

    print("Los datos han sido codificados exitosamente con one-hot.")
    print(f"La nueva forma del DataFrame es: {df_encoded.shape}")
    print("\nMuestra de los datos codificados:")
    print(df_encoded.head())



    # Probar un rango de valores de soporte para encontrar el "codo"
    support_levels = np.arange(0.01, 0.2, 0.01)
    num_itemsets = []

    print("Calculando el número de conjuntos de ítems frecuentes para diferentes niveles de soporte...")
    for support in support_levels:
        frequent_itemsets = apriori(df_encoded, min_support=support, use_colnames=True)
        num_itemsets.append(len(frequent_itemsets))
        print(f"  Soporte: {support:.2f}, Encontrados: {len(frequent_itemsets)} conjuntos de ítems frecuentes")

    # Graficar los resultados
    plt.figure(figsize=(12, 7))
    plt.plot(support_levels, num_itemsets, marker='o', linestyle='--')
    plt.title('Método del Codo para Encontrar el min_support Óptimo')
    plt.xlabel('Umbral de Soporte Mínimo')
    plt.ylabel('Número de Conjuntos de Ítems Frecuentes Encontrados')
    plt.grid(True)
    plt.xticks(support_levels)
    plt.show()


    # Muy lento. Apriori lo hace más rápido
    # # Probar un rango de valores de soporte para encontrar el "codo" (menos valores porque se demora mucho en ejecutar)
    # support_levels = np.arange(0.05, 0.21, 0.02)
    # num_itemsets = []

    # print("Calculando el número de conjuntos de ítems frecuentes para diferentes niveles de soporte...")
    # for support in support_levels:
    #     frequent_itemsets = fpgrowth(df_encoded, min_support=support, use_colnames=True)
    #     num_itemsets.append(len(frequent_itemsets))
    #     print(f"  Soporte: {support:.2f}, Encontrados: {len(frequent_itemsets)} conjuntos de ítems frecuentes")

    # # Graficar los resultados
    # plt.figure(figsize=(12, 7))
    # plt.plot(support_levels, num_itemsets, marker='o', linestyle='--')
    # plt.title('Método del Codo para Encontrar el min_support Óptimo')
    # plt.xlabel('Umbral de Soporte Mínimo')
    # plt.ylabel('Número de Conjuntos de Ítems Frecuentes Encontrados')
    # plt.grid(True)
    # plt.xticks(support_levels)
    # plt.show()



    # Min_support seleccionado
    MIN_SUPPORT = 0.05

    print(f"Ejecutando Apriori con un soporte mínimo de {MIN_SUPPORT}...")

    # 1. Encontrar conjuntos de ítems frecuentes usando Apriori
    frequent_itemsets_apriori = apriori(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)

    print("Apriori finalizado. Se encontraron", len(frequent_itemsets_apriori), "conjuntos de ítems frecuentes.")

    # 2. Generar reglas de asociación a partir de los conjuntos de ítems frecuentes
    # Filtraremos por reglas con un lift mayor a 1.2 y una confianza mayor a 0.6
    rules_apriori = association_rules(frequent_itemsets_apriori, metric="lift", min_threshold=1.2)
    rules_apriori = rules_apriori[rules_apriori['confidence'] > 0.6]


    print(f"\nSe encontraron {len(rules_apriori)} reglas significativas con Apriori.")

    # Mostrar las 10 reglas principales, ordenadas por lift (más interesantes) y luego por confianza
    print("\nTop 10 Reglas Más Significativas (Apriori):")
    print(rules_apriori.sort_values(by=['lift', 'confidence'], ascending=False).head(10))



    print(f"Ejecutando FP-Growth con un soporte mínimo de {MIN_SUPPORT}...")

    # 1. Encontrar conjuntos de ítems frecuentes usando FP-Growth
    frequent_itemsets_fpgrowth = fpgrowth(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)

    print("FP-Growth finalizado. Se encontraron", len(frequent_itemsets_fpgrowth), "conjuntos de ítems frecuentes.")

    # 2. Generar reglas de asociación a partir de los conjuntos de ítems frecuentes
    rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="lift", min_threshold=1.2)
    rules_fpgrowth = rules_fpgrowth[rules_fpgrowth['confidence'] > 0.6]

    print(f"\nSe encontraron {len(rules_fpgrowth)} reglas significativas con FP-Growth.")

    # Mostrar las 10 reglas principales, ordenadas por lift y confianza
    print("\nTop 10 Reglas Más Significativas (FP-Growth):")
    print(rules_fpgrowth.sort_values(by=['lift', 'confidence'], ascending=False).head(10))



    # --- Parte 1: Comparar los Conjuntos de Reglas de Apriori vs. FP-Growth ---
    print("="*50)
    print("Parte 1: Comparando Conjuntos de Reglas de Apriori vs. FP-Growth")
    print("="*50)
    print("Teóricamente, con el mismo soporte, ambos algoritmos deberían encontrar exactamente las mismas reglas.\nVamos a verificar esto.")

    # Para comparar, crearemos un identificador único para cada regla basado en sus ítems
    rules_apriori['rule_id'] = rules_apriori.apply(lambda row: hash(row['antecedents']) + hash(row['consequents']), axis=1)
    rules_fpgrowth['rule_id'] = rules_fpgrowth.apply(lambda row: hash(row['antecedents']) + hash(row['consequents']), axis=1)

    # Convertir los IDs de las reglas a conjuntos para una fácil comparación
    apriori_set = set(rules_apriori['rule_id'])
    fpgrowth_set = set(rules_fpgrowth['rule_id'])

    # Encontrar las diferencias
    rules_only_in_apriori = apriori_set - fpgrowth_set
    rules_only_in_fpgrowth = fpgrowth_set - apriori_set
    common_rules = apriori_set.intersection(fpgrowth_set)

    print(f"\nNúmero de reglas únicas para Apriori: {len(rules_only_in_apriori)}")
    print(f"Número de reglas únicas para FP-Growth: {len(rules_only_in_fpgrowth)}")
    print(f"Número de reglas comunes encontradas por ambos: {len(common_rules)}")

    if len(rules_only_in_apriori) == 0 and len(rules_only_in_fpgrowth) == 0:
        print("\n[ÉXITO] Como se esperaba, ambos algoritmos produjeron el conjunto idéntico de reglas.")
    else:
        print("\n[NOTA] Hay una diferencia en los conjuntos de reglas, lo cual es inesperado y puede merecer una investigación.")


    # --- Parte 2: Análisis Profundo de las Métricas de Calidad de las Reglas ---
    print("\n\n" + "="*50)
    print("Parte 2: Evaluando la Calidad de las Reglas Descubiertas")
    print("="*50)
    print("Analizaremos el conjunto de reglas final (usando los resultados de FP-Growth) basándonos en varias métricas clave.\n")

    # Usemos las reglas de FP-Growth para el análisis
    final_rules = rules_fpgrowth.drop(columns=['rule_id'])

    # --- Explicación de las Métricas ---
    print("Métricas Clave de Calidad:")
    print("  - support: La fracción de transacciones que contienen todos los ítems en la regla. (Popularidad)")
    print("  - lift: Cuánto más probable es el consecuente, dado el antecedente. (Interés). Un Lift > 1 es deseable.")
    print("  - confidence: La probabilidad de ver el consecuente en una transacción que también contiene el antecedente. (Fiabilidad)")
    print("  - conviction: Una medida de la implicación de la regla. Una convicción alta significa que el consecuente es altamente dependiente del antecedente. Es sensible a la confianza.")


    # --- Mostrando las "Mejores" Reglas ---
    print("\n--- Top 5 Reglas por LIFT (Más Interesantes/Inesperadas) ---")
    print("El Lift suele ser la mejor métrica para empezar a encontrar información accionable.")
    print(final_rules.sort_values(by='lift', ascending=False).head(5))

    print("\n--- Top 5 Reglas por CONFIDENCE (Más Confiables) ---")
    print("Estas reglas son las más propensas a ser ciertas.")
    print(final_rules.sort_values(by='confidence', ascending=False).head(5))

    print("\n--- Top 5 Reglas por CONVICTION (Implicación Más Alta) ---")
    print("Estas reglas muestran la dependencia más fuerte del consecuente sobre el antecedente.")
    print(final_rules.sort_values(by='conviction', ascending=False).head(5))



    final_rules = rules_fpgrowth.copy()

    # --- Paso 1: Definir una función para hacer las reglas legibles ---
    def format_rule(row):
        """Convierte los frozensets y métricas de una regla en una cadena de texto limpia."""
        # Unir limpiamente los ítems en el antecedente y el consecuente
        antecedent = " Y ".join([item for item in row['antecedents']])
        consequent = " Y ".join([item for item in row['consequents']])

        # Construir la declaración SI -> ENTONCES
        rule_statement = f"SI ( {antecedent} )\nENTONCES ( {consequent} )"

        # Formatear las métricas clave
        metrics = (
            f"  - Soporte: {row['support']:.4f}\n"
            f"  - Confianza: {row['confidence']:.4f}\n"
            f"  - Lift: {row['lift']:.4f}"
        )

        return f"### Regla\n{rule_statement}\n\n**Métricas:**\n{metrics}\n"

    # --- Paso 2: Construir el contenido del reporte como una cadena de texto ---
    # Empezar a construir la cadena de texto Markdown
    report_content = f"""
    # Reporte de Análisis de Reglas de Asociación

    ## Introducción
    Este reporte resume las reglas de asociación más significativas descubiertas en el conjunto de datos de homicidios en Colombia. Las reglas destacan relaciones no obvias entre diferentes atributos de un evento de homicidio.

    **Cómo Interpretar las Métricas:**
    - **Soporte:** El porcentaje de todos los homicidios donde ocurre este patrón.
    - **Confianza:** Qué tan confiable es la regla. Una confianza de 0.6 significa que si la parte 'SI' es verdadera, la parte 'ENTONCES' también es verdadera en el 60% de los casos.
    - **Lift:** Qué tan interesante es la regla. Un lift de 1.2 significa que la parte 'ENTONCES' es un 20% más probable que ocurra cuando la parte 'SI' está presente. Un lift mayor a 1 indica una relación significativa.

    ---

    """

    # --- Sección: Top 10 Reglas por Lift (Más Interesantes) ---
    report_content += "## Top 10 Reglas por Lift (Más Interesantes)\n"
    report_content += "Estas reglas representan las relaciones más inesperadas o sorprendentes en los datos.\n\n"
    top_by_lift = final_rules.sort_values(by='lift', ascending=False).head(10)
    for index, row in top_by_lift.iterrows():
        report_content += format_rule(row)
        report_content += "\n---\n"

    # --- Sección: Top 10 Reglas por Confianza (Más Confiables) ---
    report_content += "\n## Top 10 Reglas por Confianza (Más Confiables)\n"
    report_content += "Estas reglas son las más frecuentemente correctas. Si ves la condición 'SI', puedes estar muy seguro del resultado 'ENTONCES'.\n\n"
    top_by_confidence = final_rules.sort_values(by='confidence', ascending=False).head(10)
    for index, row in top_by_confidence.iterrows():
        report_content += format_rule(row)
        report_content += "\n---\n"

    # --- Paso 3: Escribir el contenido a un archivo Markdown ---
    file_name = "../data/association/reporte_reglas_asociacion.md"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[ÉXITO] ¡Reporte generado exitosamente!")

create_association_rules()