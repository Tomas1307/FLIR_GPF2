import itertools
import json
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch
import time
from datetime import datetime
import yaml

class HyperparameterSearchRecall:
    def __init__(self, 
                 dataset_paths: list,
                 target_class: int = 4, 
                 base_epochs: int = 250,
                 gpu_memory_gb: int = 24):
        
        self.dataset_paths = dataset_paths
        self.target_class = target_class
        self.base_epochs = base_epochs
        self.gpu_memory = gpu_memory_gb
        self.results = []
        
        self.output_dir = Path(f"hyperparameter_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"🔍 Búsqueda de hiperparámetros iniciada")
        print(f"🎯 Objetivo: Maximizar recall de clase {target_class} (minería ilegal)")
        print(f"📊 Datasets: {len(dataset_paths)}")
        print(f"⚡ GPU: {gpu_memory_gb}GB disponibles")
        print(f"📁 Resultados en: {self.output_dir}")

    def get_hyperparameter_combinations(self):
        """Define las combinaciones de hiperparámetros a evaluar"""
        

        
        mining_optimized_configs = [
            # Config 1: Ultra Recall - Máximo recall para minería
            #{
            #    'batch_size': 40, 'lr0': 0.02, 'lrf': 0.2, 'cos_lr': True,
            #    'weight_decay': 0.0005, 'dropout': 0.0, 'mosaic': 1.0, 
            #    'mixup': 0.15, 'imgsz': 640, 'model_size': 'yolo11m.pt',
            #    'name': 'ultra_recall'
            #},
            
            # Config 2: High Resolution - Para detalles finos
            #{
            #    'batch_size': 40, 'lr0': 0.015, 'lrf': 0.15, 'cos_lr': True,
            #    'weight_decay': 0.001, 'dropout': 0.1, 'mosaic': 0.9, 
            #    'mixup': 0.1, 'imgsz': 832, 'model_size': 'yolo11l.pt',
            #    'name': 'high_resolution'
            #},
            
            # Config 3: Balanced - Balance recall/precision
            #{
            #    'batch_size': 32, 'lr0': 0.01, 'lrf': 0.1, 'cos_lr': True,
            #    'weight_decay': 0.001, 'dropout': 0.05, 'mosaic': 0.85, 
            #    'mixup': 0.05, 'imgsz': 640, 'model_size': 'yolo11m.pt',
            #    'name': 'balanced'
            #},
            
            # Config 4: Conservative - Estable y confiable
            {
                'batch_size': 40, 'lr0': 0.005, 'lrf': 0.01, 'cos_lr': True,
                'weight_decay': 0.001, 'dropout': 0.1, 'mosaic': 0.8, 
                'mixup': 0.0, 'imgsz': 640, 'model_size': 'yolo11m.pt',
                'name': 'conservative'
            },
            
            # Config 5: Large Model Aggressive
            #{
            #    'batch_size': 32, 'lr0': 0.018, 'lrf': 0.18, 'cos_lr': False,
            #    'weight_decay': 0.002, 'dropout': 0.15, 'mosaic': 1.0, 
            #    'mixup': 0.1, 'imgsz': 640, 'model_size': 'yolo11l.pt',
            #    'name': 'large_aggressive'
            #},
            
            # Config 6: Tu configuración base mejorada
            #{
            #    'batch_size': 32, 'lr0': 0.01, 'lrf': 0.01, 'cos_lr': True,
            #    'weight_decay': 0.001, 'dropout': 0.1, 'mosaic': 0.8, 
            #    'mixup': 0.0, 'imgsz': 640, 'model_size': 'yolo11m.pt',
            #    'name': 'baseline_improved'
            #},
        ]
        
        #variations = []
        
        #base_ultra = mining_optimized_configs[0].copy()
        
        #var1 = base_ultra.copy()
        #var1.update({'batch_size': 48, 'name': 'ultra_recall_batch64'})
        #variations.append(var1)
        
        #var2 = base_ultra.copy()
        #var2.update({'lr0': 0.015, 'name': 'ultra_recall_lr_conservative'})
        #variations.append(var2)
        
        #base_hires = mining_optimized_configs[1].copy()
        
        #var3 = base_hires.copy()
        #var3.update({'model_size': 'yolo11m.pt', 'batch_size': 40, 'name': 'high_res_medium'})
        #variations.append(var3)
        
        # Combinar todas las configuraciones
        all_configs = mining_optimized_configs 
        
        print(f"Configuraciones específicas para minería ilegal: {len(all_configs)}")
        print("Enfocadas en maximizar recall de clase 4")
        
        for i, config in enumerate(all_configs):
            print(f"   {i+1}. {config['name']}: Batch={config['batch_size']}, "
                  f"LR={config['lr0']}, Img={config['imgsz']}, Model={config['model_size']}")
        
        return all_configs

    def estimate_memory_usage(self, batch_size, imgsz, model_size):
        """Estima el uso de memoria GPU"""
        base_memory = {
            'yolo11m.pt': 4.0,  # GB base
            'yolo11l.pt': 6.0   # GB base
        }
        
        memory_per_batch = batch_size * (imgsz / 640) ** 2 * 0.1
        total_memory = base_memory[model_size] + memory_per_batch
        
        return total_memory

    def train_single_config(self, config, dataset_path, run_id):
        """Entrena un modelo con una configuración específica"""
        
        estimated_memory = self.estimate_memory_usage(
            config['batch_size'], config['imgsz'], config['model_size']
        )
        
        if estimated_memory > self.gpu_memory * 0.9:
            print(f"Config {run_id}: Memoria estimada {estimated_memory:.1f}GB > {self.gpu_memory*0.9:.1f}GB")
            return None
        
        try:
            print(f"\nEntrenando configuración {run_id}:")
            print(f"   Dataset: {Path(dataset_path).name}")
            print(f"   Batch: {config['batch_size']}, LR: {config['lr0']}, Img: {config['imgsz']}")
            print(f"   Modelo: {config['model_size']}")
            
            torch.cuda.empty_cache()
            
            model = YOLO(config['model_size'])
            
            yaml_path = Path(dataset_path) / "dataset.yaml"
            config_name = config.get('name', f'config_{run_id}')
            run_name = f"hp_search_{run_id}_{config_name}_{Path(dataset_path).name}"
            
            train_args = {
                'data': str(yaml_path),
                'epochs': self.base_epochs,
                'imgsz': config['imgsz'],
                'batch': config['batch_size'],
                'lr0': config['lr0'],
                'lrf': config['lrf'],
                'cos_lr': config['cos_lr'],
                'weight_decay': config['weight_decay'],
                'dropout': config['dropout'],
                'mosaic': config['mosaic'],
                'mixup': config['mixup'],
                'patience': 5,  
                'save_period': -1, 
                'name': run_name,
                'exist_ok': True,
                'verbose': False,
                'workers': 8,
                
                'conf': 0.15,        # Confidence threshold bajo para detectar más
                'iou': 0.6,          # IoU threshold más permisivo  
                'max_det': 500,      # Más detecciones por imagen
                'augment': True,     # Data augmentation durante entrenamiento
                'agnostic_nms': False, # Class-specific NMS
                'retina_masks': True,  # Better mask quality
                'overlap_mask': True,  # Handle overlapping masks
                'mask_ratio': 4,       # Mask downsampling ratio
                'boxes': True,         # Train bounding boxes
                
                'cls': 0.5,      # Classification loss weight
                'box': 7.5,      # Box loss weight (alto para buena localización)
                'dfl': 1.5,      # Distribution focal loss weight
                'pose': 12.0,    # Pose loss weight
                'kobj': 1.0,     # Keypoint obj loss weight
                'label_smoothing': 0.0,  # Sin smoothing para preservar clase rara
                
                'hsv_h': 0.015,  # Hue augmentation (ligero)
                'hsv_s': 0.7,    # Saturation augmentation
                'hsv_v': 0.4,    # Value augmentation
                'degrees': 0.0,  # Sin rotación (minería tiene orientación específica)
                'translate': 0.1, # Translación mínima
                'scale': 0.5,    # Escala moderada
                'shear': 0.0,    # Sin shear
                'perspective': 0.0, # Sin perspectiva
                'flipud': 0.0,   # Sin flip vertical
                'fliplr': 0.5,   # Flip horizontal OK
                'copy_paste': 0.0, # Sin copy-paste
                'auto_augment': 'randaugment', # Augmentación automática
                'erasing': 0.4,  # Random erasing probability
                'crop_fraction': 1.0, # Crop fraction
            }
            
            start_time = time.time()
            results = model.train(**train_args)
            training_time = time.time() - start_time
            
            val_results = model.val(data=str(yaml_path), verbose=False)
            
            if hasattr(val_results, 'box') and val_results.box is not None:
                per_class_recall = val_results.box.r  # Recall por clase
                per_class_precision = val_results.box.p  # Precision por clase
                per_class_ap50 = val_results.box.ap50  # AP@0.5 por clase
                
                if len(per_class_recall) > self.target_class:
                    target_recall = per_class_recall[self.target_class]
                    target_precision = per_class_precision[self.target_class]
                    target_ap50 = per_class_ap50[self.target_class]
                else:
                    target_recall = target_precision = target_ap50 = 0.0
                
                overall_map50 = val_results.box.map50
                overall_map = val_results.box.map
                
            else:
                target_recall = target_precision = target_ap50 = 0.0
                overall_map50 = overall_map = 0.0
            
            result = {
                'run_id': run_id,
                'config_name': config_name,
                'dataset': Path(dataset_path).name,
                'config': config,
                'target_class_recall': float(target_recall),
                'target_class_precision': float(target_precision),
                'target_class_ap50': float(target_ap50),
                'overall_map50': float(overall_map50),
                'overall_map': float(overall_map),
                'training_time_minutes': training_time / 60,
                'final_epoch': self.base_epochs,
                'model_path': str(Path("runs/detect") / run_name / "weights" / "best.pt"),
                'gpu_memory_estimated': estimated_memory
            }
            
            print(f"   {config_name}: Recall clase {self.target_class}: {target_recall:.4f}")
            print(f"   Precision: {target_precision:.4f}, mAP@50: {overall_map50:.4f}")
            print(f"   Tiempo: {training_time/60:.1f} min, GPU: {estimated_memory:.1f}GB")
            
            return result
            
        except Exception as e:
            print(f"   Error: {str(e)}")
            return {
                'run_id': run_id,
                'dataset': Path(dataset_path).name,
                'config': config,
                'error': str(e),
                'target_class_recall': 0.0
            }
        
        finally:
            torch.cuda.empty_cache()

    def run_search(self):
        """Ejecuta la búsqueda completa de hiperparámetros"""
        
        configurations = self.get_hyperparameter_combinations()
        total_runs = len(configurations) * len(self.dataset_paths)
        
        print(f"\nINICIANDO BÚSQUEDA DE HIPERPARÁMETROS")
        print(f"   Total de experimentos: {total_runs}")
        print(f"   Tiempo estimado: {total_runs * 15:.0f} minutos")
        print("="*60)
        
        run_id = 0
        all_results = []
        
        for dataset_path in self.dataset_paths:
            print(f"\n Dataset: {Path(dataset_path).name}")
            
            for config in configurations:
                run_id += 1
                result = self.train_single_config(config, dataset_path, run_id)
                
                if result:
                    all_results.append(result)
                    
                    self.save_partial_results(all_results)
                
                time.sleep(2)
        
        self.results = all_results
        self.analyze_and_save_results()
        
        return all_results

    def save_partial_results(self, results):
        """Guarda resultados parciales"""
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "partial_results.csv", index=False)
        
        with open(self.output_dir / "partial_results.json", 'w') as f:
            json.dump(results, f, indent=2)

    def analyze_and_save_results(self):
        """Analiza y guarda los resultados finales"""
        
        df = pd.DataFrame(self.results)
        
        valid_results = df[df['target_class_recall'] > 0].copy()
        
        if len(valid_results) == 0:
            print(" No hay resultados válidos")
            return
        
        valid_results = valid_results.sort_values('target_class_recall', ascending=False)
        
        print(f"\nTOP 5 CONFIGURACIONES POR RECALL CLASE {self.target_class} (MINERÍA ILEGAL):")
        print("="*90)
        
        for i, (_, row) in enumerate(valid_results.head(5).iterrows()):
            config_name = row.get('config_name', f"Run {row['run_id']}")
            print(f"\n #{i+1} - {config_name} ({row['dataset']})")
            print(f"    Recall Minería: {row['target_class_recall']:.4f}")
            print(f"    Precision: {row['target_class_precision']:.4f}")
            print(f"    AP@50 Minería: {row['target_class_ap50']:.4f}")
            print(f"    mAP@50 General: {row['overall_map50']:.4f}")
            
            if isinstance(row['config'], dict):
                config = row['config']
                print(f"    Config: Batch={config.get('batch_size', 'N/A')}, "
                      f"LR={config.get('lr0', 'N/A')}, "
                      f"Img={config.get('imgsz', 'N/A')}, "
                      f"Model={config.get('model_size', 'N/A')}")
                print(f"    Augment: Mosaic={config.get('mosaic', 'N/A')}, "
                      f"Mixup={config.get('mixup', 'N/A')}, "
                      f"Dropout={config.get('dropout', 'N/A')}")
            
            print(f"   Tiempo: {row['training_time_minutes']:.1f} min")
            print(f"   GPU estimada: {row.get('gpu_memory_estimated', 'N/A')} GB")
        
        mining_recalls = valid_results['target_class_recall']
        print(f"\n ESTADÍSTICAS DE RECALL PARA MINERÍA ILEGAL:")
        print(f"    Mejor recall: {mining_recalls.max():.4f}")
        print(f"    Recall promedio: {mining_recalls.mean():.4f}")
        print(f"    Recall mínimo: {mining_recalls.min():.4f}")
        print(f"    Configuraciones > 0.8 recall: {len(mining_recalls[mining_recalls > 0.8])}")
        print(f"    Configuraciones > 0.9 recall: {len(mining_recalls[mining_recalls > 0.9])}")
        
        # Análisis por dataset
        print(f"\n ANÁLISIS POR DATASET:")
        for dataset in valid_results['dataset'].unique():
            subset = valid_results[valid_results['dataset'] == dataset]
            best_recall = subset['target_class_recall'].max()
            avg_recall = subset['target_class_recall'].mean()
            best_config = subset.loc[subset['target_class_recall'].idxmax(), 'config_name']
            
            print(f"    {dataset}:")
            print(f"       Mejor: {best_recall:.4f} ({best_config})")
            print(f"       Promedio: {avg_recall:.4f}")
            print(f"       Experimentos: {len(subset)}")
        
        best_overall = valid_results.iloc[0]
        print(f"\n💡 RECOMENDACIONES:")
        
        if best_overall['target_class_recall'] >= 0.9:
            print(f"   ¡EXCELENTE! Recall de {best_overall['target_class_recall']:.4f} alcanzado")
            print(f"   Usar configuración: {best_overall.get('config_name', 'N/A')}")
            print(f"   Con dataset: {best_overall['dataset']}")
        elif best_overall['target_class_recall'] >= 0.8:
            print(f"   BUENO. Recall de {best_overall['target_class_recall']:.4f}")
        else:
            print(f"   Recall bajo: {best_overall['target_class_recall']:.4f}")

        
        
        print(f"\n Resultados guardados en: {self.output_dir}")

def search_hyperparameters_for_recall():
    """Función principal para ejecutar la búsqueda"""
    
    datasets = [
    "preprocesamiento/modelo_yolov11_dataset_completo_preprocesado"
    ]
    
    searcher = HyperparameterSearchRecall(
        dataset_paths=datasets,
        target_class=4,  
        base_epochs=100,
        gpu_memory_gb=24
    )
    
    results = searcher.run_search()
    
    return searcher, results

if __name__ == "__main__":
    searcher, results = search_hyperparameters_for_recall()
    print("\n Búsqueda completada!")