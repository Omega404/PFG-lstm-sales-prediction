"""
Analisis realista de que datos UTILES podemos sacar de los 3 modelos
"""
import pandas as pd
import numpy as np
import json
import sys

# Fix encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("ANÁLISIS REALISTA: QUÉ DATOS SON REALMENTE ÚTILES")
print("=" * 80)

# ============================================================================
# 1. MODELO DE CLIENTES (V2/V3)
# ============================================================================
print("\n1. MODELO DE CLIENTES (V2/V3)")
print("-" * 80)

# Cargar métricas del modelo
with open('models/temporal/customer_v3/medium/metrics.json', 'r') as f:
    customer_metrics = json.load(f)

print("\nMÉTRICAS DEL MODELO:")
print(f"  Accuracy: {customer_metrics['purchase_prob_accuracy']:.2f}%")
print(f"  AUC: {customer_metrics['purchase_prob_auc']:.4f}")
print(f"  Days MAE: {customer_metrics['days_mae']:.2f} días")
print(f"  Value MAE: ${customer_metrics['value_mae']:.2f}")

print("\n✅ DATOS ÚTILES QUE PODEMOS SACAR:")
print("  1. PROBABILIDAD DE COMPRA (87.6% accuracy)")
print("     - Ranking de clientes por probabilidad")
print("     - Segmentación: alta/media/baja probabilidad")
print("     - Priorización de contactos")
print("     ⚠️ PERO: Valor predicho es MUY impreciso (MAE $123)")

print("\n  2. DÍAS HASTA PRÓXIMA COMPRA (5 días MAE)")
print("     - Timing de campañas")
print("     - Predicción de 'cuándo' contactar")
print("     ⚠️ PERO: MAE de 5 días en forecast de 7 días = 71% error")

print("\n  3. SEGMENTACIÓN DE CLIENTES")
print("     - Clientes activos vs inactivos")
print("     - Patrones de comportamiento")
print("     - Frecuencia de compra")

print("\n❌ DATOS NO CONFIABLES:")
print("  - Valor exacto de próxima compra ($123 MAE = ~40% error)")
print("  - Predicciones de clientes con poco historial")
print("  - Forecast de más de 7 días (fuera del alcance del modelo)")

# ============================================================================
# 2. MODELO DE PRODUCTOS
# ============================================================================
print("\n\n2. MODELO DE PRODUCTOS (SHORT/MEDIUM/LONG)")
print("-" * 80)

# Cargar métricas
with open('models/temporal/products_50epochs/short/metrics.json', 'r') as f:
    product_metrics = json.load(f)

print("\nMÉTRICAS DEL MODELO:")
print(f"  MAE: {product_metrics['mae']:.2f} unidades")
print(f"  RMSE: {product_metrics['rmse']:.2f}")
print(f"  R²: {product_metrics.get('r2', 'N/A')}")

# Cargar dataset para análisis
df = pd.read_excel('data/processed/online_retail_2.xlsx')
df = df[df['CustomerID'].notna()].copy()

# Estadísticas de productos
product_stats = df.groupby('StockCode').agg({
    'Quantity': ['sum', 'mean', 'std'],
    'Description': 'first'
}).reset_index()
product_stats.columns = ['stock_code', 'total_qty', 'mean_qty', 'std_qty', 'description']
product_stats = product_stats.sort_values('total_qty', ascending=False).head(50)

print(f"\nTop 10 productos por volumen:")
for idx, row in product_stats.head(10).iterrows():
    ratio = product_metrics['mae'] / row['mean_qty'] if row['mean_qty'] > 0 else 0
    print(f"  {row['description'][:40]:40} | Mean: {row['mean_qty']:.1f} | MAE/Mean: {ratio:.1%}")

print("\n✅ DATOS ÚTILES QUE PODEMOS SACAR:")
print("  1. PRODUCTOS DE ALTA ROTACIÓN")
print("     - Identificar productos que siempre tienen demanda")
print("     - Priorizar stock de productos populares")

print("\n  2. COMPARACIÓN RELATIVA")
print("     - Ranking de productos por demanda")
print("     - Tendencias: crece/decrece/estable")

print("\n  3. ALERTAS DE INVENTARIO")
print("     - Productos con demanda consistente")
print("     - Variabilidad de demanda (std)")

print("\n❌ DATOS NO CONFIABLES:")
print("  - Cantidad exacta (MAE 19 unidades vs mean ~10 = 190% error)")
print("  - Productos con mucha variabilidad")
print("  - Forecast de productos poco populares")
print("  ⚠️ CRÍTICO: MAE mayor que la media = modelo no sirve para valores absolutos")

# ============================================================================
# 3. ANÁLISIS CRUZADO
# ============================================================================
print("\n\n3. ANÁLISIS CRUZADO (CLIENTES + PRODUCTOS)")
print("-" * 80)

# Cargar resultados del análisis cruzado
cust = pd.read_csv('output/cross_analysis/customer_predictions_20251105_161752.csv')
rec = pd.read_csv('output/cross_analysis/recommendations_20251105_161752.csv')
inv = pd.read_csv('output/cross_analysis/inventory_forecast_20251105_161752.csv')

print("\n✅ DATOS ÚTILES QUE PODEMOS SACAR:")
print("  1. SEGMENTACIÓN COMBINADA")
high_prob = cust[cust['purchase_probability'] >= 70]
medium_prob = cust[(cust['purchase_probability'] >= 50) & (cust['purchase_probability'] < 70)]
low_prob = cust[cust['purchase_probability'] < 50]

print(f"     - Alta probabilidad (≥70%): {len(high_prob)} clientes")
print(f"     - Media probabilidad (50-70%): {len(medium_prob)} clientes")
print(f"     - Baja probabilidad (<50%): {len(low_prob)} clientes")
print(f"     → Priorizar contacto en orden de probabilidad")

print("\n  2. RANKING DE CLIENTES")
top_10 = cust.nlargest(10, 'purchase_probability')
print(f"     Top 10 clientes por probabilidad:")
for idx, row in top_10.iterrows():
    print(f"       {int(row['customer_id']):5d} | Prob: {row['purchase_probability']:5.1f}% | Histórico: ${row['total_spent_historical']:8.2f}")

print("\n  3. PRODUCTOS MÁS DEMANDADOS (RELATIVO)")
top_products = inv.nlargest(10, 'n_potential_customers')
print(f"     Productos con más clientes potenciales:")
for idx, row in top_products.iterrows():
    print(f"       {row['description'][:40]:40} | {int(row['n_potential_customers'])} clientes")

print("\n  4. PATRONES DE CROSS-SELL")
cross_sell = rec.groupby('product_description').agg({
    'customer_id': 'count',
    'is_new_product': lambda x: (x == True).sum()
}).reset_index()
cross_sell.columns = ['product', 'total_recs', 'new_customers']
cross_sell['cross_sell_rate'] = (cross_sell['new_customers'] / cross_sell['total_recs'] * 100)
cross_sell = cross_sell.sort_values('cross_sell_rate', ascending=False).head(10)
print(f"     Productos con mayor potencial de cross-sell:")
for idx, row in cross_sell.iterrows():
    print(f"       {row['product'][:40]:40} | {row['cross_sell_rate']:.1f}% nuevos")

print("\n❌ DATOS NO CONFIABLES:")
print("  - Valores monetarios exactos ($4,000 predicho vs $90 histórico = NO realista)")
print("  - Cantidades exactas de inventario (85,000 unidades = exagerado)")
print("  - ROI de 33,000% (no realista)")
print("  - Forecast de inventario en unidades absolutas")

# ============================================================================
# 4. CONCLUSIÓN: QUÉ SÍ PODEMOS USAR
# ============================================================================
print("\n\n" + "=" * 80)
print("RESUMEN: QUÉ DATOS SON REALMENTE ÚTILES")
print("=" * 80)

print("\n✅ USE CASES REALES Y CONFIABLES:")
print("\n1. SEGMENTACIÓN Y PRIORIZACIÓN DE CLIENTES")
print("   - Ranking por probabilidad de compra (confiable)")
print("   - Segmentos: alta/media/baja probabilidad")
print("   - Decidir A QUIÉN contactar")
print("   Ejemplo: 'Contactar solo clientes con prob ≥70%'")

print("\n2. TIMING DE CAMPAÑAS")
print("   - Predicción de CUÁNDO contactar (±5 días)")
print("   - Identificar clientes 'calientes' (próximos a comprar)")
print("   Ejemplo: 'Cliente #13352 predice compra en 2-7 días'")

print("\n3. PRODUCTOS POPULARES Y TRENDING")
print("   - Ranking RELATIVO de productos")
print("   - Identificar productos de alta rotación")
print("   - Comparar demanda entre productos")
print("   Ejemplo: 'JUMBO BAG tiene 2x demanda vs LUNCH BAG'")

print("\n4. RECOMENDACIONES DE CROSS-SELL")
print("   - Qué productos ofrecer a cada cliente")
print("   - Basado en historial + demanda general")
print("   Ejemplo: 'Cliente compró Jumbo Bag Red → recomendar otros colores'")

print("\n5. COMPARACIONES Y TENDENCIAS")
print("   - Comparar comportamiento entre clientes")
print("   - Tendencias de productos (crece/decrece)")
print("   - Patrones de compra por segmento")

print("\n❌ NO USAR PARA:")
print("   ✗ Valores monetarios exactos ('Cliente gastará $4,000')")
print("   ✗ Cantidades exactas de inventario ('Reabastecer 7,026 unidades')")
print("   ✗ ROI exacto de campañas")
print("   ✗ Forecast de más de 7 días")
print("   ✗ Predicciones de clientes sin historial suficiente")

print("\n" + "=" * 80)
print("RECOMENDACIÓN FINAL")
print("=" * 80)
print("""
Los modelos son buenos para:
1. RANKING y PRIORIZACIÓN (orden relativo)
2. SEGMENTACIÓN (grupos de clientes)
3. TIMING (cuándo contactar)
4. RECOMENDACIONES (qué ofrecer)

Los modelos NO son buenos para:
1. VALORES ABSOLUTOS (cantidades exactas)
2. FORECAST FINANCIERO (ingresos exactos)
3. PLANIFICACIÓN DE INVENTARIO PRECISA

Usar como HERRAMIENTA DE DECISIÓN, no como ORÁCULO.
""")

print("\n" + "=" * 80)
print("EJEMPLO DE USO CORRECTO:")
print("=" * 80)
print("""
INCORRECTO:
  "El cliente #13352 gastará exactamente $4,099.96"
  "Reabastece exactamente 7,026 unidades de JUMBO BAG"

CORRECTO:
  "El cliente #13352 tiene 100% probabilidad de compra (TOP prioridad)"
  "Contactar en próximos 2-7 días con oferta personalizada"
  "JUMBO BAG es el producto #1 en demanda (mantener stock alto)"
  "Ofrecer bundle de 3 JUMBO BAGS (otros clientes similares compraron)"
""")

print("\n" + "=" * 80)
