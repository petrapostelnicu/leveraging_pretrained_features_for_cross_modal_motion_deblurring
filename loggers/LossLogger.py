import csv


class LossLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.initialize_csv()

    def initialize_csv(self):
        with open(self.file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    def log_losses_for_epoch(self, epoch, train_loss, val_loss):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, val_loss])

    def log_all_losses(self, num_epochs, train_losses, val_losses):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            for epoch in range(num_epochs):
                writer.writerow([epoch + 1, train_losses[epoch], val_losses[epoch]])

        print(f'Losses saved to {self.file_path}')

    def log_training_time(self, time):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['total training time', time, '-'])

    def read_losses(self):
        losses = []
        with open(self.file_path, newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                epoch = int(row["epoch"])
                train_loss = float(row["train_loss"])
                val_loss = float(row["val_loss"])
                losses.append((epoch, train_loss, val_loss))
        return losses
