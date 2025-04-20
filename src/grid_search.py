import itertools
import pandas as pd
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import os


def generate_all_combinations(params):
    """Genera todas las combinaciones posibles de hiperparámetros"""
    # Generar todas las combinaciones
    keys = params.keys()
    values = params.values()
    combinations = list(itertools.product(*values))
    
    # Convertir a lista de diccionarios
    all_params = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        all_params.append(params)
    
    return all_params

def get_instance_combinations(instance_id,params, instances=8):
    """Obtiene las combinaciones para una persona específica"""
    all_combinations = generate_all_combinations(params)
    total_combinations = len(all_combinations)
    
    combinations_per_instance = total_combinations // instances
    remainder = total_combinations % instances
    
    # Ajusta el número de combinaciones si hay un remainder
    if instance_id < remainder:
        start_idx = instance_id * (combinations_per_instance + 1)
        end_idx = start_idx + combinations_per_instance + 1
    else:
        start_idx = (instance_id * combinations_per_instance) + remainder
        end_idx = start_idx + combinations_per_instance
    
    instance_combinations = all_combinations[start_idx:end_idx]
    return instance_combinations, start_idx, end_idx


def grid_search(params, instance_id, **kwargs):
    """
    Ejecuta la búsqueda de hiperparámetros para una instancia específica.

    Parámetros:
    - params: diccionario de hiperparámetros a evaluar.
    - instance_id: identificador de la instancia de la persona.

    kwargs:
    - X_train_fold: conjunto de entrenamiento.
    - y_train_fold: etiquetas de entrenamiento.
    - X_val_fold: conjunto de validación.
    - y_val_fold: etiquetas de validación.

    """
    X_train_fold = kwargs.get('X_train_fold')
    y_train_fold = kwargs.get('y_train_fold')
    X_val_fold = kwargs.get('X_val_fold')
    y_val_fold = kwargs.get('y_val_fold')
    
    print(f"Searching for instance {instance_id+1}")
    
    # Obtener combinaciones para esta persona
    param_list, start_idx, end_idx = get_instance_combinations(instance_id, params)
    print(f"Evaluando combinaciones {start_idx} a {end_idx-1} (total: {len(param_list)})")
    
    # Crear directorio para resultados si no existe
    os.makedirs('home_credit_results', exist_ok=True)
    
    try:
        print(f"✓ Data loaded - Train: {X_train_fold.shape}, Val: {X_val_fold.shape}")
        
        # Variables para seguir el mejor resultado
        best_auc = 0
        best_f1 = 0

        best_result = None
        best_params = None
        best_model = None
        results = []
        
        print(f"\nStarting process to instance: 0{instance_id+1} with {len(param_list)} combinations...")
        total_start_time = time.time()
        
        # Iterar sobre cada combinación de parámetros
        for i, params in enumerate(param_list):
            start_time = time.time()
            print(f"\n Evaluate combination: 0{i+1}/{len(param_list)} (global {start_idx+i}):")
            for param, value in params.items():
                print(f"  {param}: {value}")
            
            try:
                # Entrenar modelo con estos parámetros
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                model.fit(X_train_fold, y_train_fold)
                
                # Evaluar en conjunto de validación
                train_preds = model.predict_proba(X_train_fold)[:, 1]
                val_preds = model.predict_proba(X_val_fold)[:, 1]

                train_auc = roc_auc_score(y_train_fold, train_preds)
                val_auc = roc_auc_score(y_val_fold, val_preds)

                # Clases predichas para F1, Precision y Recall
                val_preds_class = model.predict(X_val_fold)
                val_f1 = f1_score(y_val_fold, val_preds_class)
                val_precision = precision_score(y_val_fold, val_preds_class)
                val_recall = recall_score(y_val_fold, val_preds_class)
                
                # Tiempo transcurrido
                elapsed_time = time.time() - start_time
                
                # Guardar resultados
                results.append({
                    **params,
                    'train_auc': train_auc,
                    'val_auc': val_auc,
                    'diff_auc': train_auc - val_auc,
                    'val_f1': val_f1,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'training_time': elapsed_time,
                    'global_idx': start_idx + i
                })

                print(f" Training time: {elapsed_time:.2f} seconds")
                print(f" ROC AUC Score (train): {train_auc:.4f} ")
                print(f" ROC AUC Score (Test): {val_auc:.4f} ")
                print(f" Difference AUC: {train_auc - val_auc:.4f} ")
                print(f" F1 Score: {val_f1:.4f} ")
                print(f" Precision Score: {val_precision:.4f} ")
                print(f" Recall Score: {val_recall:.4f} ")
                
                # Actualizar si se considera AUC mejor opcion

                # if val_auc > best_auc:
                #     best_auc = val_auc
                #     new_best = True
                #     print(f" New model found! (AUC: {best_auc:.4f})")
                #     best_result = {
                #         'train_auc': train_auc,
                #         'val_auc': val_auc,
                #         'diff_auc': train_auc - val_auc,
                #         'val_f1': val_f1,
                #         'val_precision': val_precision,
                #         'val_recall': val_recall
                #     }
                #     best_params = params
                #     best_model = model

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    print(f" New model found! (F1: {best_f1:.4f})")
                    best_result = {
                        'train_auc': train_auc,
                        'val_auc': val_auc,
                        'diff_auc': train_auc - val_auc,
                        'val_f1': val_f1,
                        'val_precision': val_precision,
                        'val_recall': val_recall
                    }
                    best_params = params
                    best_model = model
                    

            except Exception as e:
                print(f"  Error to evaluate combination: {str(e)}")
                results.append({
                    **params,
                    'val_auc': float('nan'),
                    'val_f1': float('nan'),
                    'val_precision': float('nan'),
                    'val_recall': float('nan'),
                    'training_time': time.time() - start_time,
                    'error': str(e),
                    'global_idx': start_idx + i
                })
        
        total_time = time.time() - total_start_time
        
        # Después de la búsqueda, mostrar el mejor resultado
        print("\n" + "="*70)
        print(f"INSTANCE RESULTS {instance_id+1} ({total_time/60:.1f} MIN)")
        print("="*70)
        
        if best_model is not None:
            print(f"  BEST TRAIN AUC: {best_result['train_auc']:.4f}")
            print(f"  BEST VAL AUC: {best_result['val_auc']:.4f}")
            print(f"  BEST DIFF AUC: {best_result['diff_auc']:.4f}")
            print(f"  BEST F1: {best_result['val_f1']:.4f}")
            print(f"  BEST Precision: {best_result['val_precision']:.4f}")
            print(f"  BEST Recall: {best_result['val_recall']:.4f}")
            print("\nBEST PARAMS:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            
            # Guardar resultados
            results_df = pd.DataFrame(results)
            results_df.to_csv(f'home_credit_results/instance{instance_id+1}_results.csv', index=False)
            
            # Guardar el mejor modelo
            with open(f'home_credit_results/instance{instance_id+1}_best_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)
            
            print(f"\n✅ PROCESS TO INSTANCE: {instance_id+1} FINISHED")
        else:
            print(f"❌ MODEL NOT FOUND {instance_id+1}.")

    except Exception as e:
        print(f"ERROR DURING MODEL VALIDATION {instance_id+1}: {str(e)}")
        import traceback
        traceback.print_exc()