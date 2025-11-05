"""
MLflow Tracking Helper - Wrapper simplificado para tracking de experimentos
"""

try:
    import mlflow
    import mlflow.keras
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowTracker:
    """Helper class para simplificar tracking con MLflow"""

    def __init__(self, experiment_name, enabled=True):
        """
        Args:
            experiment_name: Nombre del experimento (ej: 'products_temporal')
            enabled: Si False, deshabilita tracking incluso si MLflow está disponible
        """
        self.enabled = enabled and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        self.run_id = None

        if self.enabled:
            mlflow.set_experiment(experiment_name)

    def start_run(self, run_name, params=None, tags=None):
        """Inicia un run de MLflow"""
        if not self.enabled:
            return None

        mlflow.start_run(run_name=run_name)
        self.run_id = mlflow.active_run().info.run_id

        if params:
            mlflow.log_params(params)

        if tags:
            mlflow.set_tags(tags)

        return self.run_id

    def log_metrics(self, metrics, step=None):
        """Log métricas"""
        if not self.enabled:
            return

        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, file_path, artifact_path=None):
        """Log un archivo como artefacto"""
        if not self.enabled:
            return

        mlflow.log_artifact(file_path, artifact_path)

    def log_model(self, model, artifact_path, registered_model_name=None):
        """Log modelo de Keras"""
        if not self.enabled:
            return

        try:
            mlflow.keras.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name
            )
            return True
        except Exception as e:
            print(f"⚠️ Error logging modelo: {e}")
            return False

    def end_run(self):
        """Finaliza el run actual"""
        if not self.enabled:
            return

        mlflow.end_run()
        self.run_id = None

    @staticmethod
    def is_available():
        """Retorna True si MLflow está disponible"""
        return MLFLOW_AVAILABLE


# Función de conveniencia para logging de entrenamiento epoch por epoch
def log_training_history(tracker, history_dict):
    """
    Loguea el historial de entrenamiento época por época

    Args:
        tracker: Instancia de MLflowTracker
        history_dict: history.history de Keras
    """
    if not tracker.enabled:
        return

    num_epochs = len(history_dict['loss'])

    for epoch in range(num_epochs):
        epoch_metrics = {}

        for metric_name, values in history_dict.items():
            if len(values) > epoch:
                epoch_metrics[f"epoch_{metric_name}"] = values[epoch]

        tracker.log_metrics(epoch_metrics, step=epoch)
