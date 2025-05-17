from src.data_loading import load_raw_data
from src.preprocessing import preprocess_data, save_processed_data
from src.exploratory_analysis import describe_columns, plot_target_distribution
from tabulate import tabulate

def main():
    print("Cargando datos...")
    df = load_raw_data()
    
    print("\nRealizando análisis exploratorio inicial...")
    description = describe_columns(df)
    print(tabulate(description, headers='keys', tablefmt='fancy_grid'))
    
    print("\nPreprocesando datos...")
    df_processed = preprocess_data(df)
    
    print("\nGuardando datos preprocesados...")
    save_processed_data(df_processed)
    
    print("\nVisualizando distribución de la variable objetivo...")
    plot_target_distribution(df_processed)
    
    print("\nProceso completado exitosamente!")

if __name__ == "__main__":
    main()