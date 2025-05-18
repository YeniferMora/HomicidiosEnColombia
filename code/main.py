from src.data_loading import load_raw_data
from src.preprocessing import preprocess_data, save_processed_data
from src.exploratory_analysis import exploratory_analysis

def main():
    print("Cargando datos...")
    df = load_raw_data()
    
    print("\nRealizando análisis exploratorio inicial...")
    exploratory_analysis(df)
        
    print("\nPreprocesando datos...")
    df_processed = preprocess_data(df)
    
    print("\nGuardando datos preprocesados...")
    save_processed_data(df_processed)
    
    print("\nProceso completado exitosamente!")

if __name__ == "__main__":
    main()