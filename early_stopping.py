class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, restore_best_weights=True):
        """
        Early stopping to monitor validation performance and stop training when no improvement is observed.
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best value
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score, model):
        """
        Check if training should be stopped based on validation score.
        Args:
            val_score (float): Current validation score (higher is better)
            model (torch.nn.Module): Current model
        """
        if self.best_score is None:
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_model = model.state_dict().copy()
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            if self.restore_best_weights:
                self.best_model = model.state_dict().copy()
            self.counter = 0

    def restore_model(self, model):
        """
        Restore model weights from the best epoch.
        Args:
            model (torch.nn.Module): Model to restore weights to
        """
        if self.restore_best_weights and self.best_model is not None:
            model.load_state_dict(self.best_model)
