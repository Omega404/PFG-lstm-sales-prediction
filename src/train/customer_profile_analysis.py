import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ===============================
# Configuración
# ===============================
DATA_PATH = "data/processed/product_demand.xlsx"
REPORTS_DIR = "results/reports/customers"
PROFILES_DIR = "results/profiles"

# Parámetros de segmentación RFM
RFM_CONFIG = {
    'recency_segments': [7, 30, 90],      # días desde última compra
    'frequency_segments': [2, 5, 10],     # número de compras
    'monetary_segments': [50, 200, 500]   # valor total gastado
}

# Umbrales de predicción
PREDICTION_CONFIG = {
    'min_purchases_for_prediction': 3,    # Mínimo de compras para predecir
    'confidence_threshold': 0.6,          # Confianza mínima
    'days_ahead_prediction': 30           # Ventana de predicción
}

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)

# ===============================
# Clase Principal
# ===============================
class CustomerProfileAnalyzer:
    """
    Analizador de perfiles de clientes con predicción de comportamiento.
    
    Funcionalidades:
    - Análisis de patrones temporales (día, hora, frecuencia)
    - Segmentación RFM (Recency, Frequency, Monetary)
    - Identificación de productos favoritos
    - Predicción de próxima compra
    - Recomendaciones personalizadas
    """
    
    def __init__(self, data_path=DATA_PATH):
        self.data_path = data_path
        self.df = None
        self.customer_profiles = {}
        self.reference_date = None
        
    def load_data(self):
        """Carga y prepara datos de transacciones"""
        print("📂 Cargando datos de transacciones...")
        
        self.df = pd.read_excel(self.data_path, engine="openpyxl")
        self.df['StockCode'] = self.df['StockCode'].astype(str).str.strip()
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'], errors='coerce')
        
        # Agregar columnas derivadas
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['UnitPrice']
        self.df['DayOfWeek'] = self.df['InvoiceDate'].dt.dayofweek
        self.df['DayName'] = self.df['InvoiceDate'].dt.day_name()
        self.df['Hour'] = self.df['InvoiceDate'].dt.hour
        self.df['Month'] = self.df['InvoiceDate'].dt.month
        self.df['MonthName'] = self.df['InvoiceDate'].dt.month_name()
        self.df['Date'] = self.df['InvoiceDate'].dt.date
        
        # Fecha de referencia (última fecha del dataset)
        self.reference_date = self.df['InvoiceDate'].max()
        
        # Eliminar transacciones canceladas y sin CustomerID
        self.df = self.df[
            (self.df['Quantity'] > 0) & 
            (self.df['CustomerID'].notna())
        ].copy()
        
        print(f"✅ Datos cargados: {len(self.df):,} transacciones")
        print(f"   Clientes únicos: {self.df['CustomerID'].nunique():,}")
        print(f"   Período: {self.df['InvoiceDate'].min().date()} a {self.df['InvoiceDate'].max().date()}")
        print(f"   Fecha de referencia: {self.reference_date.date()}\n")
        
        return self.df
    
    def analyze_temporal_patterns(self, customer_id):
        """
        Analiza patrones temporales de un cliente.
        
        Returns:
            dict con patrones detectados
        """
        customer_data = self.df[self.df['CustomerID'] == customer_id].copy()
        
        if len(customer_data) == 0:
            return None
        
        # Patrón por día de semana
        day_counts = customer_data['DayName'].value_counts()
        preferred_day = day_counts.index[0]
        day_concentration = (day_counts.iloc[0] / len(customer_data) * 100)
        
        # Patrón horario
        hour_counts = customer_data['Hour'].value_counts()
        preferred_hour = hour_counts.index[0]
        
        # Clasificar horario
        if 6 <= preferred_hour < 12:
            time_period = "Mañana"
        elif 12 <= preferred_hour < 18:
            time_period = "Tarde"
        elif 18 <= preferred_hour < 22:
            time_period = "Noche"
        else:
            time_period = "Madrugada"
        
        # Patrón mensual
        month_counts = customer_data['MonthName'].value_counts()
        preferred_months = month_counts.head(3).index.tolist()
        
        # Frecuencia de compra
        unique_dates = customer_data['Date'].unique()
        if len(unique_dates) >= 2:
            dates_sorted = sorted(unique_dates)
            intervals = [(dates_sorted[i+1] - dates_sorted[i]).days 
                        for i in range(len(dates_sorted)-1)]
            avg_days_between = np.mean(intervals)
            std_days_between = np.std(intervals) if len(intervals) > 1 else 0
        else:
            avg_days_between = None
            std_days_between = None
        
        return {
            'preferred_day': preferred_day,
            'day_concentration_pct': float(day_concentration),
            'preferred_hour': int(preferred_hour),
            'time_period': time_period,
            'preferred_months': preferred_months,
            'avg_days_between_purchases': float(avg_days_between) if avg_days_between else None,
            'purchase_frequency_std': float(std_days_between) if std_days_between else None,
            'is_regular_buyer': bool(std_days_between < 7 if std_days_between else False)
        }
    
    def calculate_rfm_score(self, customer_id):
        """
        Calcula score RFM (Recency, Frequency, Monetary).
        
        Returns:
            dict con scores y segmento
        """
        customer_data = self.df[self.df['CustomerID'] == customer_id]
        
        # Recency: días desde última compra
        last_purchase = customer_data['InvoiceDate'].max()
        recency_days = (self.reference_date - last_purchase).days
        
        # Frequency: número de compras únicas
        frequency = customer_data['InvoiceNo'].nunique()
        
        # Monetary: valor total gastado
        monetary = customer_data['TotalPrice'].sum()
        
        # Asignar scores (1-5, donde 5 es mejor)
        def get_score(value, segments, reverse=False):
            segments_sorted = sorted(segments)
            for i, threshold in enumerate(segments_sorted):
                if value <= threshold:
                    score = i + 1 if not reverse else len(segments_sorted) - i + 1
                    return score
            return len(segments_sorted) + 1 if not reverse else 1
        
        r_score = get_score(recency_days, RFM_CONFIG['recency_segments'], reverse=True)
        f_score = get_score(frequency, RFM_CONFIG['frequency_segments'])
        m_score = get_score(monetary, RFM_CONFIG['monetary_segments'])
        
        # Segmento RFM
        rfm_segment = f"{r_score}{f_score}{m_score}"
        
        # Clasificación del cliente
        if r_score >= 4 and f_score >= 4 and m_score >= 4:
            customer_type = "VIP"
        elif r_score >= 4 and f_score >= 3:
            customer_type = "Leal"
        elif r_score >= 4 and f_score <= 2:
            customer_type = "Nuevo Prometedor"
        elif r_score <= 2 and f_score >= 3:
            customer_type = "En Riesgo"
        elif r_score <= 2 and f_score <= 2:
            customer_type = "Perdido"
        else:
            customer_type = "Regular"
        
        return {
            'recency_days': int(recency_days),
            'frequency_purchases': int(frequency),
            'monetary_total': float(monetary),
            'r_score': int(r_score),
            'f_score': int(f_score),
            'm_score': int(m_score),
            'rfm_segment': rfm_segment,
            'customer_type': customer_type
        }
    
    def analyze_product_preferences(self, customer_id):
        """
        Analiza preferencias de productos del cliente.
        
        Returns:
            dict con productos favoritos y patrones
        """
        customer_data = self.df[self.df['CustomerID'] == customer_id]
        
        # Productos más comprados
        product_counts = customer_data.groupby('StockCode').agg({
            'Quantity': 'sum',
            'TotalPrice': 'sum',
            'InvoiceNo': 'nunique',
            'Description': 'first'
        }).reset_index()
        
        product_counts.columns = ['StockCode', 'TotalQuantity', 'TotalSpent', 
                                   'PurchaseCount', 'Description']
        product_counts = product_counts.sort_values('PurchaseCount', ascending=False)
        
        # Top productos
        top_products = []
        for _, row in product_counts.head(10).iterrows():
            top_products.append({
                'stock_code': row['StockCode'],
                'description': str(row['Description'])[:50],
                'purchase_count': int(row['PurchaseCount']),
                'total_quantity': int(row['TotalQuantity']),
                'total_spent': float(row['TotalSpent'])
            })
        
        # Métricas de compra
        avg_basket_size = customer_data.groupby('InvoiceNo')['Quantity'].sum().mean()
        avg_basket_value = customer_data.groupby('InvoiceNo')['TotalPrice'].sum().mean()
        unique_products = customer_data['StockCode'].nunique()
        
        return {
            'top_products': top_products,
            'unique_products_bought': int(unique_products),
            'avg_basket_size': float(avg_basket_size),
            'avg_basket_value': float(avg_basket_value),
            'favorite_product': top_products[0] if top_products else None
        }
    
    def predict_next_purchase(self, customer_id):
        """
        Predice próxima compra del cliente.
        
        Returns:
            dict con predicción
        """
        customer_data = self.df[self.df['CustomerID'] == customer_id].sort_values('InvoiceDate')
        
        if len(customer_data['Date'].unique()) < PREDICTION_CONFIG['min_purchases_for_prediction']:
            return {
                'can_predict': False,
                'reason': 'Insuficiente historial (mínimo 3 compras)'
            }
        
        # Calcular intervalos entre compras
        unique_dates = sorted(customer_data['Date'].unique())
        intervals = [(unique_dates[i+1] - unique_dates[i]).days 
                    for i in range(len(unique_dates)-1)]
        
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        
        # Última compra
        last_purchase_date = customer_data['InvoiceDate'].max()
        days_since_last = (self.reference_date - last_purchase_date).days
        
        # Predicción de próxima compra
        expected_next_purchase = last_purchase_date + timedelta(days=avg_interval)
        days_until_next = (expected_next_purchase - self.reference_date).days
        
        # Probabilidad basada en el patrón
        if std_interval > 0:
            z_score = abs(days_since_last - avg_interval) / std_interval
            probability = max(0, 1 - (z_score / 3))  # 3 sigmas
        else:
            probability = 0.8 if days_since_last <= avg_interval else 0.3
        
        # Productos probables (basado en frecuencia)
        product_freq = customer_data['StockCode'].value_counts().head(5)
        likely_products = []
        for stock_code, count in product_freq.items():
            desc = customer_data[customer_data['StockCode'] == stock_code]['Description'].iloc[0]
            likely_products.append({
                'stock_code': stock_code,
                'description': str(desc)[:50],
                'purchase_frequency': int(count),
                'probability': float(count / len(customer_data['InvoiceNo'].unique()))
            })
        
        # Valor esperado de compra
        historical_values = customer_data.groupby('InvoiceNo')['TotalPrice'].sum()
        expected_value = historical_values.mean()
        
        return {
            'can_predict': True,
            'last_purchase_date': last_purchase_date.strftime('%Y-%m-%d'),
            'days_since_last_purchase': int(days_since_last),
            'expected_next_purchase_date': expected_next_purchase.strftime('%Y-%m-%d'),
            'days_until_expected_purchase': int(days_until_next),
            'purchase_probability': float(probability),
            'avg_days_between_purchases': float(avg_interval),
            'purchase_regularity_score': float(1 / (std_interval + 1)),  # Más regular = mayor score
            'likely_products': likely_products,
            'expected_purchase_value': float(expected_value)
        }
    
    def create_customer_profile(self, customer_id):
        """
        Crea perfil completo de un cliente.
        
        Returns:
            dict con perfil completo
        """
        customer_data = self.df[self.df['CustomerID'] == customer_id]
        
        if len(customer_data) == 0:
            return None
        
        # Información básica
        first_purchase = customer_data['InvoiceDate'].min()
        last_purchase = customer_data['InvoiceDate'].max()
        total_transactions = customer_data['InvoiceNo'].nunique()
        total_spent = customer_data['TotalPrice'].sum()
        total_items = customer_data['Quantity'].sum()
        
        # Análisis componentes
        temporal_patterns = self.analyze_temporal_patterns(customer_id)
        rfm_scores = self.calculate_rfm_score(customer_id)
        product_prefs = self.analyze_product_preferences(customer_id)
        prediction = self.predict_next_purchase(customer_id)
        
        # Compilar perfil
        profile = {
            'customer_id': str(customer_id),
            'profile_generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            
            'summary': {
                'first_purchase_date': first_purchase.strftime('%Y-%m-%d'),
                'last_purchase_date': last_purchase.strftime('%Y-%m-%d'),
                'customer_lifetime_days': int((last_purchase - first_purchase).days),
                'total_transactions': int(total_transactions),
                'total_spent': float(total_spent),
                'total_items_purchased': int(total_items),
                'avg_transaction_value': float(total_spent / total_transactions),
                'avg_items_per_transaction': float(total_items / total_transactions)
            },
            
            'temporal_patterns': temporal_patterns,
            'rfm_analysis': rfm_scores,
            'product_preferences': product_prefs,
            'prediction': prediction
        }
        
        return profile
    
    def generate_personalized_offer(self, customer_id):
        """
        Genera oferta personalizada para un cliente.
        
        Returns:
            dict con oferta y justificación
        """
        profile = self.customer_profiles.get(customer_id) or self.create_customer_profile(customer_id)
        
        if not profile:
            return None
        
        rfm = profile['rfm_analysis']
        prediction = profile['prediction']
        temporal = profile['temporal_patterns']
        prefs = profile['product_preferences']
        
        # Lógica de oferta basada en tipo de cliente
        customer_type = rfm['customer_type']
        
        if customer_type == "VIP":
            discount = 20
            message = "¡Gracias por ser un cliente VIP! Oferta exclusiva para ti."
            urgency = "BAJA"
        elif customer_type == "Leal":
            discount = 15
            message = "Apreciamos tu lealtad. Descuento especial en tus favoritos."
            urgency = "MEDIA"
        elif customer_type == "En Riesgo":
            discount = 25
            message = "¡Te extrañamos! Vuelve con este descuento especial."
            urgency = "ALTA"
        elif customer_type == "Perdido":
            discount = 30
            message = "¡Recuperemos tu confianza! Gran descuento de bienvenida."
            urgency = "ALTA"
        elif customer_type == "Nuevo Prometedor":
            discount = 10
            message = "Sigue descubriendo nuestros productos con este descuento."
            urgency = "MEDIA"
        else:
            discount = 10
            message = "Oferta especial para ti."
            urgency = "BAJA"
        
        # Productos recomendados
        if prefs['top_products']:
            recommended_products = [p['stock_code'] for p in prefs['top_products'][:3]]
        else:
            recommended_products = []
        
        # Momento óptimo de contacto
        optimal_day = temporal['preferred_day']
        optimal_time = temporal['time_period']
        
        offer = {
            'customer_id': customer_id,
            'customer_type': customer_type,
            'discount_percent': discount,
            'message': message,
            'urgency': urgency,
            'recommended_products': recommended_products,
            'optimal_contact_day': optimal_day,
            'optimal_contact_time': optimal_time,
            'valid_until': (datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
            'expected_response_rate': float(prediction.get('purchase_probability', 0) if prediction.get('can_predict') else 0.5)
        }
        
        return offer
    
    def analyze_customers_batch(self, customer_ids=None, top_n=None):
        """
        Analiza múltiples clientes en batch.
        
        Args:
            customer_ids: Lista específica de IDs o None para todos
            top_n: Si None, analizar solo top N clientes por gasto
        """
        print(f"\n{'='*70}")
        print(f"👥 ANÁLISIS DE PERFILES DE CLIENTES")
        print(f"{'='*70}\n")
        
        if customer_ids is None:
            if top_n:
                # Top N clientes por gasto total
                top_customers = self.df.groupby('CustomerID')['TotalPrice'].sum().nlargest(top_n)
                customer_ids = top_customers.index.tolist()
                print(f"📊 Analizando top {top_n} clientes por gasto total\n")
            else:
                customer_ids = self.df['CustomerID'].unique()
                print(f"📊 Analizando TODOS los clientes ({len(customer_ids)})\n")
        
        profiles = []
        offers = []
        
        for i, customer_id in enumerate(customer_ids, 1):
            if i % 10 == 0 or i == len(customer_ids):
                print(f"   Procesando: {i}/{len(customer_ids)} clientes...")
            
            try:
                # Crear perfil
                profile = self.create_customer_profile(customer_id)
                if profile:
                    self.customer_profiles[customer_id] = profile
                    profiles.append(profile)
                    
                    # Generar oferta
                    offer = self.generate_personalized_offer(customer_id)
                    if offer:
                        offers.append(offer)
                        
            except Exception as e:
                print(f"   ⚠️  Error con cliente {customer_id}: {str(e)}")
                continue
        
        # Guardar resultados
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Perfiles
        profiles_path = os.path.join(PROFILES_DIR, f'customer_profiles_{timestamp}.json')
        with open(profiles_path, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, ensure_ascii=False)
        
        # Ofertas
        offers_path = os.path.join(REPORTS_DIR, f'personalized_offers_{timestamp}.json')
        with open(offers_path, 'w', encoding='utf-8') as f:
            json.dump(offers, f, indent=2, ensure_ascii=False)
        
        # Resumen Excel
        self._generate_summary_excel(profiles, offers, timestamp)
        
        print(f"\n{'='*70}")
        print(f"📊 RESUMEN DEL ANÁLISIS")
        print(f"{'='*70}")
        print(f"Clientes procesados: {len(profiles)}")
        print(f"Ofertas generadas: {len(offers)}")
        
        if profiles:
            # Estadísticas RFM
            customer_types = [p['rfm_analysis']['customer_type'] for p in profiles]
            type_counts = Counter(customer_types)
            
            print(f"\n📈 Distribución por tipo:")
            for ctype, count in type_counts.most_common():
                pct = count / len(profiles) * 100
                print(f"   {ctype}: {count} ({pct:.1f}%)")
            
            # Estadísticas de ofertas
            if offers:
                avg_discount = np.mean([o['discount_percent'] for o in offers])
                urgency_counts = Counter([o['urgency'] for o in offers])
                
                print(f"\n💰 Ofertas:")
                print(f"   Descuento promedio: {avg_discount:.1f}%")
                for urgency, count in urgency_counts.items():
                    print(f"   Urgencia {urgency}: {count}")
        
        print(f"\n📁 Archivos generados:")
        print(f"   - {profiles_path}")
        print(f"   - {offers_path}")
        print(f"\n✅ Proceso completado!")
        
        return profiles, offers
    
    def _generate_summary_excel(self, profiles, offers, timestamp):
        """Genera resumen en Excel"""
        summary_data = []
        
        for profile in profiles:
            cid = profile['customer_id']
            offer = next((o for o in offers if str(o['customer_id']) == cid), None)
            
            row = {
                'CustomerID': cid,
                'CustomerType': profile['rfm_analysis']['customer_type'],
                'TotalSpent': profile['summary']['total_spent'],
                'TotalTransactions': profile['summary']['total_transactions'],
                'LastPurchase': profile['summary']['last_purchase_date'],
                'PreferredDay': profile['temporal_patterns']['preferred_day'],
                'PreferredTime': profile['temporal_patterns']['time_period'],
                'RFM_Segment': profile['rfm_analysis']['rfm_segment'],
                'R_Score': profile['rfm_analysis']['r_score'],
                'F_Score': profile['rfm_analysis']['f_score'],
                'M_Score': profile['rfm_analysis']['m_score']
            }
            
            if offer:
                row['OfferDiscount'] = offer['discount_percent']
                row['OfferUrgency'] = offer['urgency']
                row['ExpectedResponse'] = offer['expected_response_rate']
            
            summary_data.append(row)
        
        df_summary = pd.DataFrame(summary_data)
        excel_path = os.path.join(REPORTS_DIR, f'customer_summary_{timestamp}.xlsx')
        df_summary.to_excel(excel_path, index=False)
        
        print(f"   - {excel_path}")

# ===============================
# Ejecución principal
# ===============================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("👥 SISTEMA DE ANÁLISIS DE PERFILES DE CLIENTES")
    print("="*70 + "\n")
    
    # Inicializar analizador
    analyzer = CustomerProfileAnalyzer()
    
    # Cargar datos
    analyzer.load_data()
    
    # Analizar top 20 clientes por gasto
    profiles, offers = analyzer.analyze_customers_batch(top_n=20)
    
    print("\n✅ Sistema de perfiles de clientes completado!")
