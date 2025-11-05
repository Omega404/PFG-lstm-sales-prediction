import pandas as pd
import numpy as np

# Cargar archivos
cust = pd.read_csv(r'output/cross_analysis/customer_predictions_20251105_161752.csv')
rec = pd.read_csv(r'output/cross_analysis/recommendations_20251105_161752.csv')
inv = pd.read_csv(r'output/cross_analysis/inventory_forecast_20251105_161752.csv')

print("=" * 80)
print("INSIGHTS DE NEGOCIO - ANÁLISIS CRUZADO")
print("=" * 80)

# 1. Análisis de conversión predicha
print("\n1. ANÁLISIS DE CONVERSIÓN Y POTENCIAL DE INGRESOS")
print("-" * 80)

high_value = cust[cust['segment'] == 'high_value_high_prob']
medium_prob = cust[cust['segment'] == 'medium_prob']
low_prob = cust[cust['segment'] == 'low_prob']

print(f"\nSegmento HIGH VALUE (4 clientes):")
print(f"  - Probabilidad media: {high_value['purchase_probability'].mean():.1f}%")
print(f"  - Valor total predicho: ${high_value['predicted_value'].sum():,.2f}")
print(f"  - Gasto histórico: ${high_value['total_spent_historical'].sum():,.2f}")
potencial_hv = (high_value['predicted_value'].sum() / high_value['total_spent_historical'].sum() - 1) * 100
print(f"  - Potencial de crecimiento: {potencial_hv:.1f}%")

print(f"\nSegmento MEDIUM PROB (8 clientes):")
print(f"  - Probabilidad media: {medium_prob['purchase_probability'].mean():.1f}%")
print(f"  - Valor total predicho: ${medium_prob['predicted_value'].sum():,.2f}")

print(f"\nSegmento LOW PROB (88 clientes):")
print(f"  - Probabilidad media: {low_prob['purchase_probability'].mean():.1f}%")
print(f"  - NO contactar (bajo ROI)")

# 2. Análisis de inventario
print("\n\n2. FORECAST DE INVENTARIO (próximos 7 días)")
print("-" * 80)

total_units = inv['total_forecast'].sum()
total_value = (inv['total_forecast'] * inv['price']).sum()

print(f"\nDemanda total estimada: {total_units:,.0f} unidades")
print(f"Valor estimado del inventario: ${total_value:,.2f}")
print(f"\nTop 5 productos críticos a reabastecer:")
top5_inv = inv.head(5)
for idx, row in top5_inv.iterrows():
    print(f"  - {row['description'][:40]:40} | {row['total_forecast']:,} unidades | {row['n_potential_customers']} clientes")

# 3. Análisis de productos cross-sell
print("\n\n3. OPORTUNIDADES DE CROSS-SELL")
print("-" * 80)

# Productos más recomendados
prod_recs = rec.groupby(['stock_code', 'product_description']).agg({
    'customer_id': 'count',
    'recommendation_score': 'mean',
    'is_new_product': lambda x: x.sum()
}).reset_index()
prod_recs.columns = ['stock_code', 'product', 'n_recommendations', 'avg_score', 'new_customers']
prod_recs = prod_recs.sort_values('n_recommendations', ascending=False).head(10)

print(f"\nProductos con mayor potencial de cross-sell:")
for idx, row in prod_recs.iterrows():
    is_new = f"({row['new_customers']} nuevos)" if row['new_customers'] > 0 else "(recompra)"
    print(f"  - {row['product'][:40]:40} | {row['n_recommendations']:2d} clientes {is_new}")

# 4. ROI estimado de campaña
print("\n\n4. ESTIMACIÓN DE ROI DE CAMPAÑA")
print("-" * 80)

# Solo contactar high_value + medium_prob
campaign_customers = len(high_value) + len(medium_prob)
expected_revenue = high_value['predicted_value'].sum() + medium_prob['predicted_value'].sum()

# Asumiendo costos de campaña
cost_per_contact = 5  # $5 por contacto (email + seguimiento)
campaign_cost = campaign_customers * cost_per_contact

print(f"\nClientes a contactar: {campaign_customers} (high_value + medium_prob)")
print(f"Costo de campaña estimado: ${campaign_cost:,.2f}")
print(f"Ingresos esperados (predichos): ${expected_revenue:,.2f}")
print(f"ROI estimado: {(expected_revenue / campaign_cost - 1) * 100:.1f}%")

# 5. Análisis de clientes específicos
print("\n\n5. CASOS ESPECÍFICOS DE ALTO VALOR")
print("-" * 80)

top_3_customers = cust.nlargest(3, 'predicted_value')
print("\nTop 3 clientes a priorizar:")
for idx, customer in top_3_customers.iterrows():
    print(f"\n  Cliente #{int(customer['customer_id'])}:")
    print(f"    - Probabilidad: {customer['purchase_probability']:.1f}%")
    print(f"    - Valor predicho: ${customer['predicted_value']:,.2f}")
    print(f"    - Histórico: ${customer['total_spent_historical']:.2f} en {int(customer['n_purchases_historical'])} compras")
    print(f"    - Días desde última compra: {int(customer['days_since_last'])} días")

    # Recomendaciones para este cliente
    customer_recs = rec[rec['customer_id'] == customer['customer_id']].nlargest(3, 'recommendation_score')
    if len(customer_recs) > 0:
        print(f"    - Productos recomendados:")
        for _, rec_row in customer_recs.iterrows():
            print(f"      * {rec_row['product_description'][:50]}")

# 6. Comparación con histórico
print("\n\n6. COMPARACIÓN CON COMPORTAMIENTO HISTÓRICO")
print("-" * 80)

print(f"\nAnálisis de todos los clientes analizados (100):")
print(f"  Gasto histórico total: ${cust['total_spent_historical'].sum():,.2f}")
print(f"  Gasto predicho (próximos 7d): ${cust['predicted_value'].sum():,.2f}")
print(f"  Tasa de actividad semanal: {(cust['predicted_value'].sum() / cust['total_spent_historical'].sum()) * 100:.2f}%")

print("\n" + "=" * 80)
print("FIN DEL ANÁLISIS")
print("=" * 80)
