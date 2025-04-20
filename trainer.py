import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn

class MAMLTrainer:
    def __init__(self, model, device, inner_lr=0.01, meta_lr=0.001):
        self.model = model.to(device)
        self.device = device
        self.inner_lr = inner_lr
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def adapt(self, task, num_steps=5):
        fast_weights = dict(self.model.named_parameters())
        for _ in range(num_steps):
            outputs = self.model(task['support_x'], fast_weights)
            loss = self.loss_fn(outputs, task['support_y'])
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = {
                name: param - self.inner_lr * grad
                for (name, param), grad in zip(fast_weights.items(), grads)
            }
        return fast_weights

    def evaluate(self, task, params=None):
        with torch.no_grad():
            outputs = self.model(task['query_x'], params)
            loss = self.loss_fn(outputs, task['query_y'])
            preds = outputs.argmax(dim=1)
            accuracy = (preds == task['query_y']).float().mean()
        return loss.item(), accuracy.item()

    def train_epoch(self, dataset, num_tasks=100):
        self.model.train()
        total_loss, total_acc = 0, 0
        successful_tasks = 0
        
        pbar = tqdm(range(num_tasks), desc="Training")
        for _ in pbar:
            try:
                task = dataset.get_task_batch()
                task = {k: v.to(self.device) for k, v in task.items()}
                fast_weights = self.adapt(task)
                
                self.meta_optimizer.zero_grad()
                outputs = self.model(task['query_x'], fast_weights)
                loss = self.loss_fn(outputs, task['query_y'])
                loss.backward()
                self.meta_optimizer.step()
                
                preds = outputs.argmax(dim=1)
                acc = (preds == task['query_y']).float().mean().item()
                
                total_loss += loss.item()
                total_acc += acc
                successful_tasks += 1
                
                pbar.set_postfix({"loss": loss.item(), "acc": acc})
            except Exception as e:
                print(f"Error during training task: {e}")
                continue
                
        if successful_tasks == 0:
            raise ValueError("No tasks were successfully processed during training")
            
        return total_loss / successful_tasks, total_acc / successful_tasks

    def validate(self, dataset, num_tasks=50):
        self.model.eval()
        total_loss, total_acc = 0, 0
        successful_tasks = 0
        
        pbar = tqdm(range(num_tasks), desc="Validation")
        for _ in pbar:
            try:
                task = dataset.get_task_batch()
                task = {k: v.to(self.device) for k, v in task.items()}
                fast_weights = self.adapt(task)
                loss, acc = self.evaluate(task, fast_weights)
                
                total_loss += loss
                total_acc += acc
                successful_tasks += 1
                
                pbar.set_postfix({"loss": loss, "acc": acc})
            except Exception as e:
                print(f"Error during validation task: {e}")
                continue
        
        if successful_tasks == 0:
            raise ValueError("No tasks were successfully processed during validation")
            
        return total_loss / successful_tasks, total_acc / successful_tasks
