import os
import subprocess
import time
from queue import PriorityQueue
from threading import Thread, Lock

# Define a lock for thread-safe GPU allocation
gpu_lock = Lock()

# Define job structure
class GPUJob:
    def __init__(self, job_id, priority, command, gpu_count=1):
        self.job_id = job_id
        self.priority = priority  # Lower number = higher priority
        self.command = command
        self.gpu_count = gpu_count

    def __lt__(self, other):
        return self.priority < other.priority


class GPUScheduler:
    def __init__(self):
        self.queue = PriorityQueue()  # Priority queue for job scheduling
        self.active_jobs = []  # List of currently running jobs

    def add_job(self, job):
        self.queue.put(job)
        print(f"[INFO] Job {job.job_id} added to the queue with priority {job.priority}")

    def monitor_gpus(self):
        """Monitor GPU utilization and availability."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                      '--format=csv,nounits,noheader'], 
                                     capture_output=True, text=True, check=True)
            gpu_stats = result.stdout.strip().split('\n')
            gpus = []
            for idx, gpu in enumerate(gpu_stats):
                util, mem_used, mem_total = map(int, gpu.split(', '))
                gpus.append({
                    'id': idx,
                    'utilization': util,
                    'memory_used': mem_used,
                    'memory_total': mem_total,
                    'available_memory': mem_total - mem_used
                })
            return gpus
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to retrieve GPU stats: {e}")
            return []

    def allocate_gpus(self, job):
        """Allocate GPUs for the given job."""
        available_gpus = self.monitor_gpus()
        allocated_gpus = []
        for gpu in available_gpus:
            if len(allocated_gpus) < job.gpu_count and gpu['utilization'] < 50:
                allocated_gpus.append(gpu['id'])

        if len(allocated_gpus) == job.gpu_count:
            print(f"[INFO] Job {job.job_id} assigned GPUs: {allocated_gpus}")
            return allocated_gpus
        else:
            print(f"[INFO] Not enough GPUs available for Job {job.job_id}")
            return None

    def run_job(self, job):
        """Run a job on allocated GPUs."""
        with gpu_lock:
            allocated_gpus = self.allocate_gpus(job)
            if allocated_gpus:
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, allocated_gpus))
                try:
                    print(f"[INFO] Starting Job {job.job_id}")
                    process = subprocess.Popen(job.command, shell=True)
                    self.active_jobs.append((job, process))
                    process.wait()
                    print(f"[INFO] Job {job.job_id} completed")
                except Exception as e:
                    print(f"[ERROR] Job {job.job_id} failed: {e}")
                finally:
                    self.active_jobs.remove((job, process))
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                print(f"[INFO] Job {job.job_id} re-queued due to insufficient GPUs")
                self.add_job(job)

    def schedule_jobs(self):
        """Continuously schedule jobs from the queue."""
        while True:
            if not self.queue.empty():
                job = self.queue.get()
                thread = Thread(target=self.run_job, args=(job,))
    

