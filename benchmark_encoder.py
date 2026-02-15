import torch
import time
from cissn.models.encoder import DisentangledStateEncoder

def benchmark_encoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on {device}")
    
    batch_size = 64
    seq_len = 96
    input_dim = 1
    
    model = DisentangledStateEncoder(input_dim=input_dim).to(device)
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    # Warmup
    for _ in range(10):
        _ = model(x)
        
    start_time = time.time()
    n_iters = 100
    for _ in range(n_iters):
        _ = model(x)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    avg_time = (end_time - start_time) / n_iters
    
    print(f"Average Forward Pass Time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {batch_size / avg_time:.2f} samples/sec")

if __name__ == "__main__":
    benchmark_encoder()
