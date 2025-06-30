# Mathematics Thesis

## Resumen

Este trabajo aborda la compactación y eficiencia de las **funciones de historial** de máquinas de Turing. Partiendo de la motivación de la implicación  
\[
  \textsf{P} = \textsf{NP} \;\implies\; \textsf{P} = \textsf{PSPACE},
\]
se investiga si estas funciones admiten descripciones de tamaño polinómico y evaluación en tiempo polinómico. Para ello se combinan resultados teóricos con una amplia experimentación numérica.

## Resultados Teóricos

- **Restricción de paridad fija**: Se muestra una transformación polinómica que convierte cualquier MT en otra cuyas configuraciones siempre tienen la misma paridad de bits, forzando una Forma Normal Disyuntiva (FND) de tamaño exponencial en el peor caso.  
- **Desglose monopaso**: A través de una construcción polinómica se define una _pseudo_-MT equivalente en la que configuraciones consecutivas difieren en un solo bit, recuperando así posibilidades de simplificación lógica y dejando abierta la viabilidad de descripciones compactas.

## Resultados Experimentales

- Se generaron miles de LBAs aleatorias y se midieron sus funciones de historial con métricas de **enredo** (*entanglement*) y **ecuanimidad** (*equanimity*), obteniendo valores consistentemente bajos.  
- Las FND mínimas de los historiales resultaron muy inferiores en términos de número de literales y términos comparadas con funciones booleanas aleatorias del mismo tamaño.  
- Al proyectar los historiales sobre 5 bits de la cinta, el 75 % de las funciones coinciden con circuitos de hasta 10 puertas, indicando compacidad práctica.  
- Aunque se forzaron ejecuciones muy largas (contadores binarios completos y alternantes), la complejidad lógica de las FND se mantuvo moderada, confirmando la robustez de la hipótesis.

## Cómo citar este trabajo

Si utilizas resultados de esta tesis, por favor cítala así:

```bibtex
@mastersthesis{Garcia-Lopez2025,
  author       = {Pablo García López},
  title        = {{Estudio de la complejidad de las funciones booleanas de historial de máquinas de Turing}},
  school       = {Universidad Complutense de Madrid},
  year         = {2025},
  address      = {Madrid, España},
  url          = {https://github.com/pabgarcialopez/Math-thesis},
}
