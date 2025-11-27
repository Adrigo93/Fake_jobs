
# Detección de ofertas de trabajo falsas (NLP con PyTorch)

Proyecto personal de NLP junior.  

El proyecto es que a partir del texto de una oferta de trabajo, el modelo intente decir si es real o fake.

---

## ¿Qué he hecho?

- He usado el dataset público de Kaggle Real or Fake Job Posting Prediction.
- He unido varias columnas de texto (título, descripción, requisitos, etc.) en un solo campo `text`.
- He representado el texto con TF-IDF (hasta 5.000 palabras) usando `scikit-learn`.
- Encima de esos vectores TF-IDF he entrenado una red neuronal pequeña en PyTorch:
  - `Linear(5000 → 256) + ReLU + Dropout + Linear(256 → 2)`  
  - Salidas: `real` / `fake`.

---

## Resultados

- Accuracy en validación: alrededor de 98%.
- Ha tenido un buen rendimiento detectando ofertas reales y un buen rendimiento razonable en las ofertas falsas.

En la carpeta del proyecto se guardan:

- `fake_jobs_loss_curve.png` – curva de pérdida (train/val).
- `fake_jobs_acc_curve.png` – curva de accuracy (train/val).
- `fake_jobs_confusion_matrix.png` – matriz de confusión (real vs fake).

---

Cosas que me gustaría mejorar

- Probar a manejar mejor el desbalanceo de clases (pesos de clase u oversampling).
- Testear modelos de texto más avanzados (por ejemplo, algún Transformer tipo BERT).
- Guardar modelo + vectorizador para poder pasarle ofertas nuevas y ver la predicción.

Es un proyecto sencillo, pero me ha servido para practicar el flujo completo de NLP con PyTorch:
desde el CSV hasta las métricas y las gráficas finales.
