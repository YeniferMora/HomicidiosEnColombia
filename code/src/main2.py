
from data_loading import load_raw_data
from preprocessing import preprocess_data, save_processed_data
from exploratory_analysis import exploratory_analysis
from association_rules import create_association_rules
from config import PROCESSED_DATA_PATH
from grouping_k import ClusteringHomicidios
from nationalGrouping import NationalGrouping

def main():
    print("Cargando datos...")
    df = load_raw_data()
    
    print("\nRealizando análisis exploratorio inicial...")
    exploratory_analysis(df)
        
    print("\nPreprocesando datos...")
    df_processed = preprocess_data(df)
    
    print("\nGuardando datos preprocesados...")
    save_processed_data(df_processed)

    print("\nReglas de asociación...")
    create_association_rules()
    print("\nReglas de asociación creadas en data/association/reporte_reglas_asociacion.md...")
    
    print("\nProceso completado exitosamente!")
    
    print("\nAgrupación nacional...")
    ng = NationalGrouping(PROCESSED_DATA_PATH)
    ng.load_and_balance_data()
    X_scaled = ng.preprocess()
    ng.cluster_dbscan(X_scaled)
    ng.cluster_gmm(X_scaled)
    
    print("\nAgrupación nacional completada exitosamente!")
    
    # ejecutar grouping_k.py
        
    clustering = ClusteringHomicidios(df)
    resultados = clustering.ejecutar_pipeline_completo()
    print("\nAgrupación K completada exitosamente!")
    print(resultados)

if __name__ == "__main__":
    main()