from src.data_loading import load_raw_data
from src.preprocessing import preprocess_data, save_processed_data
from src.exploratory_analysis import exploratory_analysis
from src.association_rules import create_association_rules

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

if __name__ == "__main__":
    main()