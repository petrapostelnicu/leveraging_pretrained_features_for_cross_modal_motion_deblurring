import csv


class EvaluationLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.initialize_csv()

    def initialize_csv(self):
        with open(self.file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Metric', 'Result'])

    def log_psnr(self, psnr):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['PSNR', psnr])

    def log_ssim(self, ssim):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['SSIM', ssim])

    def log_inference_time(self, time):
        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['inference_time', time])
