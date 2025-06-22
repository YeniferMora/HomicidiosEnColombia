
    # Reporte de Análisis de Reglas de Asociación

    ## Introducción
    Este reporte resume las reglas de asociación más significativas descubiertas en el conjunto de datos de homicidios en Colombia. Las reglas destacan relaciones no obvias entre diferentes atributos de un evento de homicidio.

    **Cómo Interpretar las Métricas:**
    - **Soporte:** El porcentaje de todos los homicidios donde ocurre este patrón.
    - **Confianza:** Qué tan confiable es la regla. Una confianza de 0.6 significa que si la parte 'SI' es verdadera, la parte 'ENTONCES' también es verdadera en el 60% de los casos.
    - **Lift:** Qué tan interesante es la regla. Un lift de 1.2 significa que la parte 'ENTONCES' es un 20% más probable que ocurra cuando la parte 'SI' está presente. Un lift mayor a 1 indica una relación significativa.

    ---

    ## Top 10 Reglas por Lift (Más Interesantes)
Estas reglas representan las relaciones más inesperadas o sorprendentes en los datos.

### Regla
SI ( Escenario del Hecho_vía pública Y Mecanismo Causal_Corto punzante )
ENTONCES ( Ancestro Racial_Mestizo Y Sexo de la victima_Hombre Y Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0661
  - Confianza: 0.6921
  - Lift: 1.3745

---
### Regla
SI ( Escenario del Hecho_vía pública Y Sexo de la victima_Hombre Y Mecanismo Causal_Corto punzante )
ENTONCES ( Ancestro Racial_Mestizo Y Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0661
  - Confianza: 0.7317
  - Lift: 1.3349

---
### Regla
SI ( Escenario del Hecho_vía pública Y Mecanismo Causal_Corto punzante )
ENTONCES ( Ancestro Racial_Mestizo Y Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0697
  - Confianza: 0.7299
  - Lift: 1.3315

---
### Regla
SI ( Escenario del Hecho_vía pública Y Ancestro Racial_Mestizo Y Mecanismo Causal_Corto punzante )
ENTONCES ( Sexo de la victima_Hombre Y Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0661
  - Confianza: 0.8544
  - Lift: 1.3187

---
### Regla
SI ( Escenario del Hecho_vía pública Y Mecanismo Causal_Corto punzante )
ENTONCES ( Zona del Hecho_Cabecera municipal Y Sexo de la victima_Hombre )

**Métricas:**
  - Soporte: 0.0813
  - Confianza: 0.8513
  - Lift: 1.3138

---
### Regla
SI ( Escenario del Hecho_vía pública Y Grupo de edad de la victima_(20 a 24) )
ENTONCES ( Mecanismo Causal_Proyectil de arma de fuego Y Sexo de la victima_Hombre Y Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0700
  - Confianza: 0.6238
  - Lift: 1.3046

---
### Regla
SI ( Ancestro Racial_Mestizo Y Mecanismo Causal_Proyectil de arma de fuego Y Zona del Hecho_Cabecera municipal Y Grupo de edad de la victima_(20 a 24) )
ENTONCES ( Escenario del Hecho_vía pública Y Sexo de la victima_Hombre )

**Métricas:**
  - Soporte: 0.0532
  - Confianza: 0.6380
  - Lift: 1.3005

---
### Regla
SI ( Mecanismo Causal_Proyectil de arma de fuego Y Zona del Hecho_Cabecera municipal Y Grupo de edad de la victima_(20 a 24) )
ENTONCES ( Escenario del Hecho_vía pública Y Sexo de la victima_Hombre )

**Métricas:**
  - Soporte: 0.0700
  - Confianza: 0.6377
  - Lift: 1.3000

---
### Regla
SI ( Sexo de la victima_Hombre Y Zona del Hecho_Cabecera municipal Y Ancestro Racial_Mestizo Y Mecanismo Causal_Proyectil de arma de fuego Y Grupo de edad de la victima_(20 a 24) )
ENTONCES ( Escenario del Hecho_vía pública )

**Métricas:**
  - Soporte: 0.0532
  - Confianza: 0.6722
  - Lift: 1.2994

---
### Regla
SI ( Escenario del Hecho_vía pública Y Ancestro Racial_Mestizo Y Grupo de edad de la victima_(20 a 24) )
ENTONCES ( Mecanismo Causal_Proyectil de arma de fuego Y Sexo de la victima_Hombre Y Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0532
  - Confianza: 0.6198
  - Lift: 1.2963

---

## Top 10 Reglas por Confianza (Más Confiables)
Estas reglas son las más frecuentemente correctas. Si ves la condición 'SI', puedes estar muy seguro del resultado 'ENTONCES'.

### Regla
SI ( Escenario del Hecho_vía pública Y Ancestro Racial_Mestizo Y Sexo de la victima_Hombre Y Mecanismo Causal_Corto punzante )
ENTONCES ( Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0661
  - Confianza: 0.9025
  - Lift: 1.2817

---
### Regla
SI ( Escenario del Hecho_vía pública Y Ancestro Racial_Mestizo Y Mecanismo Causal_Corto punzante )
ENTONCES ( Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0697
  - Confianza: 0.9011
  - Lift: 1.2797

---
### Regla
SI ( Escenario del Hecho_vía pública Y Sexo de la victima_Hombre Y Mecanismo Causal_Corto punzante )
ENTONCES ( Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0813
  - Confianza: 0.9000
  - Lift: 1.2782

---
### Regla
SI ( Escenario del Hecho_vía pública Y Mecanismo Causal_Corto punzante )
ENTONCES ( Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0859
  - Confianza: 0.8986
  - Lift: 1.2761

---
### Regla
SI ( Escenario del Hecho_vía pública Y Rango de Hora del Hecho X 3 Horas_Sin información Y Mecanismo Causal_Corto punzante )
ENTONCES ( Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0504
  - Confianza: 0.8956
  - Lift: 1.2719

---
### Regla
SI ( Escenario del Hecho_vía pública Y Ancestro Racial_Mestizo Y Mecanismo Causal_Corto punzante )
ENTONCES ( Sexo de la victima_Hombre Y Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0661
  - Confianza: 0.8544
  - Lift: 1.3187

---
### Regla
SI ( Escenario del Hecho_vía pública Y Mecanismo Causal_Corto punzante )
ENTONCES ( Zona del Hecho_Cabecera municipal Y Sexo de la victima_Hombre )

**Métricas:**
  - Soporte: 0.0813
  - Confianza: 0.8513
  - Lift: 1.3138

---
### Regla
SI ( Escenario del Hecho_vía pública Y Ancestro Racial_Mestizo Y Grupo de edad de la victima_(20 a 24) )
ENTONCES ( Sexo de la victima_Hombre Y Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0690
  - Confianza: 0.8048
  - Lift: 1.2421

---
### Regla
SI ( Escenario del Hecho_vía pública Y Grupo de edad de la victima_(20 a 24) )
ENTONCES ( Sexo de la victima_Hombre Y Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0898
  - Confianza: 0.8010
  - Lift: 1.2363

---
### Regla
SI ( Escenario del Hecho_vía pública Y Ancestro Racial_Mestizo Y Grupo de edad de la victima_(25 a 29) )
ENTONCES ( Sexo de la victima_Hombre Y Zona del Hecho_Cabecera municipal )

**Métricas:**
  - Soporte: 0.0632
  - Confianza: 0.7958
  - Lift: 1.2282

---
