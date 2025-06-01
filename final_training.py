import json
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
from datetime import datetime

class ConservativeFinalTraining:
    def __init__(self):
        # Configuración ganadora directamente codificada
        self.best_config = {
            'model_size': 'yolo11m.pt',
            'batch_size': 40,
            'lr0': 0.005,          # CLAVE: LR conservador
            'lrf': 0.01,
            'cos_lr': True,
            'weight_decay': 0.001,
            'dropout': 0.1,        # CLAVE: Regularización ligera
            'mosaic': 0.8,         # CLAVE: No máximo
            'mixup': 0.0,          # CLAVE: Sin mixup
            'imgsz': 640,
            'name': 'conservative_winner'
        }
        
        print(f"🏆 CONFIGURACIÓN GANADORA CARGADA:")
        print(f"   🥇 Recall alcanzado: 86.25% (mejor de todos)")
        print(f"   📁 Dataset: preprocesado (CLAHE + reducción ruido)")
        print(f"   ⚙️ Configuración conservative:")
        for key, value in self.best_config.items():
            print(f"      {key}: {value}")
    
    def prepare_conservative_training_config(self, 
                                           dataset_path: str,
                                           epochs: int = 100,
                                           patience: int = 20,
                                           save_period: int = 25):
        """Prepara la configuración final conservative optimizada"""
        
        config = {
                'batch_size': 40, 'lr0': 0.005, 'lrf': 0.01, 'cos_lr': True,
                'weight_decay': 0.001, 'dropout': 0.1, 'mosaic': 0.8, 
                'mixup': 0.0, 'imgsz': 640, 'model_size': 'yolo11m.pt',
                'name': 'conservative'
                }
        
        run_name = f"final_training_conservative_{Path(dataset_path).name}"
        # Configuración conservative específica para minería ilegal
        # ESTO es lo que sabemos que funciona (86.25% recall):
        train_args = {
                'data': str(Path(dataset_path)/'dataset.yaml'),
                'epochs': 100,
                'imgsz': config['imgsz'],
                'batch': config['batch_size'],
                'lr0': config['lr0'],
                'lrf': config['lrf'],
                'cos_lr': config['cos_lr'],
                'weight_decay': config['weight_decay'],
                'dropout': config['dropout'],
                'mosaic': config['mosaic'],
                'mixup': config['mixup'],
                'patience': 3,  # Early stopping
                'save_period': -1,  # No guardar checkpoints intermedios
                'name': run_name,
                'exist_ok': True,
                'verbose': False,
                'workers': 16,
                
                # 🎯 CONFIGURACIONES ESPECÍFICAS PARA RECALL DE MINERÍA
                'conf': 0.15,        # Confidence threshold bajo para detectar más
                'iou': 0.6,          # IoU threshold más permisivo  
                'max_det': 500,      # Más detecciones por imagen
                'augment': True,     # Data augmentation durante entrenamiento
                'agnostic_nms': False, # Class-specific NMS
                'retina_masks': True,  # Better mask quality
                'overlap_mask': True,  # Handle overlapping masks
                'mask_ratio': 4,       # Mask downsampling ratio
                'boxes': True,         # Train bounding boxes
                
                # Optimizaciones de pérdida para clase desbalanceada
                'cls': 0.5,      # Classification loss weight
                'box': 7.5,      # Box loss weight (alto para buena localización)
                'dfl': 1.5,      # Distribution focal loss weight
                'pose': 12.0,    # Pose loss weight
                'kobj': 1.0,     # Keypoint obj loss weight
                'label_smoothing': 0.0,  # Sin smoothing para preservar clase rara
                
                # Augmentaciones específicas para minería
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
        
        return train_args, self.best_config['model_size']
    
    def train_conservative_model(self, 
                                dataset_path: str,
                                output_name: str = "conservative_mining_detector",
                                epochs: int = 100,
                                resume_from: str = None):
        """Entrena el modelo final con la configuración conservative ganadora"""
        
        print(f"\nENTRENAMIENTO CONSERVATIVE FINAL")
        print(f"Objetivo: Superar 86.25% recall de minería ilegal")
        print(f"Dataset: {dataset_path}")
        print(f"Épocas: {epochs}")
        print(f"Configuración: Conservative (rank #1)")
        print("="*60)
        
        # Verificar dataset
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
        
        # Preparar configuración
        train_args, model_weights = self.prepare_conservative_training_config(
            dataset_path, epochs
        )
        
        # Crear directorio de salida con timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_dir = Path(f"conservative_final_{output_name}_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        # Nombre del run
        run_name = f"conservative_final_{Path(dataset_path).name}"
        train_args['name'] = run_name
        train_args['project'] = str(output_dir)
        
        try:
            # Limpiar cache GPU
            torch.cuda.empty_cache()
            
            # Crear o cargar modelo
            if resume_from and Path(resume_from).exists():
                print(f"🔄 Reanudando desde: {resume_from}")
                model = YOLO(resume_from)
            else:
                print(f"🆕 Creando YOLOv11m desde: {model_weights}")
                model = YOLO(model_weights)
            
            # Mostrar configuración clave
            print(f"\nCONFIGURACIÓN CONSERVATIVE:")
            print(f"   Batch Size: {train_args['batch']}")
            print(f"   Learning Rate: {train_args['lr0']} (conservador)")
            print(f"   Mosaic: {train_args['mosaic']} (no máximo)")
            print(f"   Mixup: {train_args['mixup']} (desactivado)")
            print(f"   Dropout: {train_args['dropout']} (regularización ligera)")
            print(f"   Patience: {train_args['patience']} (early stopping extendido)")
            
            # Entrenar
            print(f"\n⚡ Iniciando entrenamiento conservative...")
            print(f"🎯 Meta: >90% recall en minería ilegal")
            
            results = model.train(**train_args)
            
            # Cargar mejor modelo
            best_model_path = output_dir / run_name / "weights" / "best.pt"
            if not best_model_path.exists():
                raise FileNotFoundError(f"Modelo entrenado no encontrado: {best_model_path}")
            
            final_model = YOLO(str(best_model_path))
            
            # Evaluación final detallada con thresholds optimizados para recall
            print(f"\n📊 EVALUACIÓN FINAL CON THRESHOLDS OPTIMIZADOS")
            
            # Evaluación con threshold bajo para máximo recall
            val_results_recall = final_model.val(
                data=train_args['data'],
                imgsz=train_args['imgsz'],
                batch=train_args['batch'],
                conf=0.1,          # Threshold muy bajo para recall
                iou=0.5,           # IoU permisivo
                max_det=1000,
                verbose=True,
                plots=True,
                save_json=True
            )
            
            # Evaluación con threshold normal para comparación
            val_results_normal = final_model.val(
                data=train_args['data'],
                imgsz=train_args['imgsz'],
                batch=train_args['batch'],
                conf=0.25,         # Threshold normal
                iou=0.7,           # IoU estándar
                max_det=300,
                verbose=False
            )
            
            # Extraer métricas
            mining_metrics_recall = self.extract_mining_metrics(val_results_recall, "optimizado_recall")
            mining_metrics_normal = self.extract_mining_metrics(val_results_normal, "normal")
            
            # Comparar con resultado original (86.25%)
            baseline_recall = 0.8625
            improvement = mining_metrics_recall['recall'] - baseline_recall
            
            print(f"\nRESULTADOS FINALES - MINERÍA ILEGAL (CLASE 4):")
            print("="*60)
            print(f"🎯 THRESHOLD OPTIMIZADO PARA RECALL (conf=0.1):")
            print(f"   Recall: {mining_metrics_recall['recall']:.4f} ({mining_metrics_recall['recall']*100:.2f}%)")
            print(f"   Precision: {mining_metrics_recall['precision']:.4f}")
            print(f"   AP@50: {mining_metrics_recall['ap50']:.4f}")
            
            print(f"\n📊 THRESHOLD NORMAL (conf=0.25):")
            print(f"   Recall: {mining_metrics_normal['recall']:.4f} ({mining_metrics_normal['recall']*100:.2f}%)")
            print(f"   Precision: {mining_metrics_normal['precision']:.4f}")
            print(f"   AP@50: {mining_metrics_normal['ap50']:.4f}")
            
            print(f"\n📈 COMPARACIÓN CON BASELINE:")
            print(f"   Baseline (20 épocas): {baseline_recall:.4f} (86.25%)")
            print(f"   Final ({epochs} épocas): {mining_metrics_recall['recall']:.4f}")
            print(f"   Mejora: {improvement:+.4f} ({improvement*100:+.2f} puntos %)")
            
            if mining_metrics_recall['recall'] >= 0.90:
                print(f"   ✅ ¡META ALCANZADA! Recall ≥ 90%")
            elif mining_metrics_recall['recall'] >= 0.87:
                print(f"   🎯 ¡EXCELENTE!baseline superado")
            else:
                print(f"   Por debajo del baseline, considera más entrenamiento")
            
            # Guardar métricas completas
            all_metrics = {
                'config_used': self.best_config,
                'dataset': dataset_path,
                'training_epochs': epochs,
                'model_path': str(best_model_path),
                'recall_optimized': mining_metrics_recall,
                'normal_threshold': mining_metrics_normal,
                'baseline_comparison': {
                    'baseline_recall': baseline_recall,
                    'final_recall': mining_metrics_recall['recall'],
                    'improvement': improvement
                },
                'recommendations': self.generate_recommendations(mining_metrics_recall)
            }
            
            # Guardar resultados
            metrics_file = output_dir / "conservative_mining_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(all_metrics, f, indent=2)
            
            # Copiar modelo final con nombre descriptivo
            final_model_path = output_dir / f"conservative_mining_detector_recall_{mining_metrics_recall['recall']:.3f}.pt"
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            
            print(f"\nENTRENAMIENTO CONSERVATIVE COMPLETADO")
            print(f" Modelo final: {final_model_path}")
            print(f" Métricas: {metrics_file}")
            print(f" Directorio completo: {output_dir}")
            
            return final_model, all_metrics
            
        except Exception as e:
            print(f" Error durante el entrenamiento: {str(e)}")
            raise e
        
        finally:
            torch.cuda.empty_cache()
    
    def extract_mining_metrics(self, val_results, eval_type):
        """Extrae métricas específicas de minería ilegal"""
        if hasattr(val_results, 'box') and val_results.box is not None:
            per_class_recall = val_results.box.r
            per_class_precision = val_results.box.p
            per_class_ap50 = val_results.box.ap50
            
            # Métricas de minería ilegal (clase 4)
            if len(per_class_recall) > 4:
                return {
                    'eval_type': eval_type,
                    'recall': float(per_class_recall[4]),
                    'precision': float(per_class_precision[4]),
                    'ap50': float(per_class_ap50[4]),
                    'overall_map50': float(val_results.box.map50),
                    'overall_map': float(val_results.box.map)
                }
        
        return {
            'eval_type': eval_type,
            'recall': 0.0, 'precision': 0.0, 'ap50': 0.0,
            'overall_map50': 0.0, 'overall_map': 0.0
        }
    
    def generate_recommendations(self, metrics):
        """Genera recomendaciones basadas en los resultados"""
        recommendations = []
        
        recall = metrics['recall']
        precision = metrics['precision']
        
        if recall >= 0.95:
            recommendations.append("🏆 Excelente recall! Modelo listo para producción")
        elif recall >= 0.90:
            recommendations.append("✅ Muy buen recall! Considera ajustar threshold para balance recall/precision")
        elif recall >= 0.85:
            recommendations.append("🎯 Buen recall. Considera entrenamiento adicional o threshold más bajo")
        else:
            recommendations.append("⚠️ Recall bajo. Considera más datos o técnicas de data augmentation")
        
        if precision < 0.5:
            recommendations.append("📊 Precision baja. Considera threshold más alto en producción")
        elif precision < 0.7:
            recommendations.append("📈 Precision moderada. Balance aceptable para detección de minería")
        else:
            recommendations.append("🎯 Excelente precision. Buen balance recall/precision")
        
        return recommendations

# Funciones de uso fácil
def train_conservative_mining_detector(dataset_path: str = "preprocesamiento/modelo_yolov11_dataset_completo_preprocesado",
                                      epochs: int = 100,
                                      resume_from: str = None):
    """
    Función principal para entrenar el detector conservative de minería ilegal
    
    Args:
        dataset_path: Ruta al dataset (default: preprocesado)
        epochs: Número de épocas (default: 100)
        resume_from: Ruta a modelo para resumir entrenamiento
    """
    
    print("🏆 ENTRENADOR CONSERVATIVE DE MINERÍA ILEGAL")
    print("📊 Basado en la configuración ganadora (86.25% recall)")
    print("="*60)
    
    # Crear trainer
    trainer = ConservativeFinalTraining()
    
    # Verificar dataset
    if not Path(dataset_path).exists():
        print(f"❌Dataset no encontrado: {dataset_path}")
        
        # Ofrecer alternativa
        alt_path = "modelo_yolov11_dataset_completo"
        if Path(alt_path).exists():
            response = input(f"¿Usar dataset original ({alt_path})? [y/N]: ")
            if response.lower() == 'y':
                dataset_path = alt_path
            else:
                return None, None
        else:
            print(" Ningún dataset encontrado")
            return None, None
    
    # Entrenar
    try:
        final_model, metrics = trainer.train_conservative_model(
            dataset_path=dataset_path,
            output_name="conservative_winner",
            epochs=epochs,
            resume_from=resume_from
        )
        
        return final_model, metrics
        
    except KeyboardInterrupt:
        print("\n⏹ Entrenamiento interrumpido por el usuario")
        return None, None
    except Exception as e:
        print(f"\n❌Error: {str(e)}")
        return None, None

# Función de análisis post-entrenamiento
def analyze_conservative_results(metrics_file: str):
    """Analiza los resultados del entrenamiento conservative"""
    
    if not Path(metrics_file).exists():
        print(f" Archivo de métricas no encontrado: {metrics_file}")
        return
    
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    print("📊 ANÁLISIS DE RESULTADOS CONSERVATIVE")
    print("="*50)
    
    recall_opt = metrics['recall_optimized']
    normal = metrics['normal_threshold']
    baseline = metrics['baseline_comparison']
    
    print(f"🎯 RENDIMIENTO FINAL:")
    print(f"   Mejor Recall: {recall_opt['recall']:.4f} ({recall_opt['recall']*100:.1f}%)")
    print(f"   Precision: {recall_opt['precision']:.4f}")
    print(f"   Mejora vs Baseline: {baseline['improvement']:+.4f}")
    
    print(f"\n💡 RECOMENDACIONES:")
    for rec in metrics['recommendations']:
        print(f"   {rec}")

if __name__ == "__main__":
    # Ejecutar entrenamiento conservative
    print("🚀 Iniciando entrenamiento con configuración conservative ganadora...")
    
    final_model, metrics = train_conservative_mining_detector(
        epochs=100,  # Cambiar según necesites
        resume_from=None  # O ruta a modelo para resumir
    )
    
    if metrics:
        print(f"\n🎉 ¡Entrenamiento completado!")
        recall = metrics['recall_optimized']['recall']
        print(f"🎯 Recall final: {recall:.4f} ({recall*100:.1f}%)")