import torch
import torch.nn as nn
import numpy as np
import gc
import logging
import os
from tqdm import tqdm
import queue
import threading

# Configure logger
logger = logging.getLogger(__name__)


def thomson_random_project_loss_unrolled(weight, pd=40, pn=20, pnd=0):
    """Thomson loss with random projections for intermediate layers"""
    n_input = weight.shape[1] # in_features
    n_filt = weight.shape[0] # out_features
    
    pd1 = pd
    pd2 = n_input
    
    # Calculate number of projections
    if pnd == 0:
        total_p = pn
    else:
        total_p = n_filt // pnd
        
    total_loss = 0
    
    # Generate multiple random projections
    for i in range(total_p):
        # filt = weight.view(-1, n_filt)
        filt = weight.t()
        
        # Create random projection matrix (not learnable)
        p = torch.normal(
            mean=0.0,
            std=1.0,
            size=(pd1, pd2),
            device=filt.device,
            requires_grad=False,
            generator=torch.Generator(device=filt.device).manual_seed(n_input + i),
            dtype=filt.dtype
        )
        
        # Project filters
        projected_filt = torch.mm(p, filt)
        
        # Add negative versions
        filt_neg = -projected_filt
        projected_filt = torch.cat((projected_filt, filt_neg), dim=1)
        n_filt_doubled = 2 * n_filt
        
        # Calculate cosine similarities
        filt_norm = torch.sqrt(torch.sum(projected_filt * projected_filt, dim=0, keepdim=True) + 1e-4)
        norm_mat = torch.mm(filt_norm.t(), filt_norm)
        inner_pro = torch.mm(projected_filt.t(), projected_filt)
        cos_sim = inner_pro / norm_mat
        
        # Calculate repulsion loss
        # cross_terms = 2.0 - 2.0 * cos_sim + torch.eye(n_filt_doubled, device=filt.device)
        cross_terms = 2.0 - 2.0 * cos_sim
        cross_terms.diagonal(dim1=-2, dim2=-1).add_(1.0)
        final = cross_terms.pow(-1)
        final = final.triu(diagonal=1)
        cnt = n_filt_doubled * (n_filt_doubled - 1) / 2.0
        loss = final.sum() / cnt
        
        total_loss += loss
        
    return total_loss / total_p


def thomson_random_project_loss(weight, pd=40, pn=20, pnd=0):
    """Thomson loss with random projections for intermediate layers"""
    n_input = weight.shape[1] # in_features
    n_filt = weight.shape[0] # out_features
    
    pd1 = pd
    pd2 = n_input
    
    # Calculate number of projections
    if pnd == 0:
        total_p = pn
    else:
        total_p = n_filt // pnd
        
    total_loss = 0
    
    # Helper function for checkpointing
    def run_projection_loss(w, seed_val):
        # Re-create generator for deterministic behavior during re-computation
        gen = torch.Generator(device=w.device).manual_seed(seed_val)
        
        filt = w.t() # Use .t() as corrected
        
        # Create random projection matrix
        p = torch.normal(
            mean=0.0,
            std=1.0,
            size=(pd1, pd2),
            device=filt.device,
            requires_grad=False,
            generator=gen,
            dtype=filt.dtype
        )
        
        # Project filters
        projected_filt = torch.mm(p, filt)
        
        # Add negative versions
        filt_neg = -projected_filt
        projected_filt = torch.cat((projected_filt, filt_neg), dim=1)
        n_filt_doubled = 2 * n_filt
        
        # Calculate cosine similarities
        filt_norm = torch.sqrt(torch.sum(projected_filt * projected_filt, dim=0, keepdim=True) + 1e-4)
        norm_mat = torch.mm(filt_norm.t(), filt_norm)
        inner_pro = torch.mm(projected_filt.t(), projected_filt)
        cos_sim = inner_pro / norm_mat
        
        # Calculate repulsion loss
        cross_terms = 2.0 - 2.0 * cos_sim
        cross_terms.diagonal(dim1=-2, dim2=-1).add_(1.0)
        final = cross_terms.pow(-1)
        final = final.triu(diagonal=1)
        cnt = n_filt_doubled * (n_filt_doubled - 1) / 2.0
        loss = final.sum() / cnt
        
        return loss

    # Generate multiple random projections
    for i in range(total_p):
        # Use checkpoint to save memory
        # We pass a seed so the random matrix 'p' is identical in forward and backward passes
        seed = n_input + i
        
        # Checkpointing requires the input (weight) to have requires_grad=True
        if weight.requires_grad:
             curr_loss = checkpoint(run_projection_loss, weight, seed, use_reentrant=False)
        else:
             curr_loss = run_projection_loss(weight, seed)
             
        total_loss = total_loss + curr_loss
        
    return total_loss / total_p


def calculate_total_mhe(model, target_modules_list=["attn", "mlp"]):
    mhe_losses = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and any(key in name for key in target_modules_list):
                weight = module.weight.data
                loss = mhe_loss(weight)
                mhe_losses.append(loss.cpu().item())
    return float(np.sum(mhe_losses))


def mhe_loss(filt):
    n_filt, _ = filt.shape
    filt = torch.transpose(filt, 0, 1)
    filt_neg = filt * (-1)
    filt = torch.cat((filt, filt_neg), dim=1)
    n_filt *= 2

    filt_norm = torch.sqrt(torch.sum(filt * filt, dim=0, keepdim=True) + 1e-4)
    norm_mat = torch.matmul(filt_norm.t(), filt_norm)
    inner_pro = torch.matmul(filt.t(), filt)
    inner_pro /= norm_mat

    cross_terms = (2.0 - 2.0 * inner_pro + torch.diag(torch.ones(n_filt, device=filt.device)))
    final = torch.pow(cross_terms, -0.5 * torch.ones_like(cross_terms))
    final -= torch.tril(final)
    cnt = n_filt * (n_filt - 1) / 2.0
    MHE_loss = torch.sum(final) / cnt
    return MHE_loss


def mhe_optimized_init_multithread_not_good(model):    
    with torch.no_grad():
        # First just normalize and calculate MHE loss
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            module.weight.data = module.weight.data / torch.norm(module.weight.data, dim=1, keepdim=True)
        
    # Calculate MHE loss after normalization only
    # normalized_mhe_loss = calculate_total_mhe(model)
    # print(f"MHE loss after normalization: {normalized_mhe_loss:.8f}")

    target_layers = []
    for module_name, module in model.named_modules():
        if "lm_head" in module_name:
            continue
        if isinstance(module, nn.Linear):
            target_layers.append((module_name, module))

    # Optimize each target linear layer in parallel
    import concurrent.futures
    import contextlib

    # Detect available devices
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        devices = [f'cuda:{i}' for i in range(num_gpus)]
    else:
        num_gpus = 0
        devices = ['cpu']

    def process_layer(args):
        module_name, module, target_device_str = args
        
        # Target device where optimization will happen
        target_device = torch.device(target_device_str)
        
        # Use a separate stream for each thread to allow kernel overlap on GPU
        if target_device.type == 'cuda':
            stream = torch.cuda.Stream(device=target_device)
            ctx = torch.cuda.stream(stream)
        else:
            stream = None
            ctx = contextlib.nullcontext()

        with ctx:
            # Move weight to target device for optimization
            # We detach to avoid autograd tracking on the original parameter
            weight = module.weight.detach().to(target_device).contiguous()
            weight = weight / weight.norm(dim=1, keepdim=True)
            
            # optimize_layer_mhe runs in the current stream context on target_device
            # Note: optimize_layer_mhe logs to stdout, so output might be interleaved
            optimized_fp32 = optimize_layer_mhe_fast(weight.to(torch.float32))

            # Copy result back to the module's original device
            module.weight.data.copy_(optimized_fp32.to(module.weight.device, dtype=module.weight.dtype))
            
            # Cleanup to save memory for other workers
            del weight, optimized_fp32
        
        # Wait for this stream to finish
        if stream:
            stream.synchronize()
        
        return module_name

    # Adjust max_workers based on your VRAM and number of devices. 
    # Each worker creates full-size gradients and optimizer states.
    workers_per_device = 2 if num_gpus > 0 else 4
    max_workers = max(1, num_gpus * workers_per_device)
    if num_gpus == 0: max_workers = 4
    
    print(f"Optimizing layers with {max_workers} parallel workers on {len(devices)} devices ({devices})...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks, assigning devices round-robin
        futures = []
        for i, item in enumerate(target_layers):
            assigned_device = devices[i % len(devices)]
            futures.append(executor.submit(process_layer, (*item, assigned_device)))
        
        # Process results as they complete to update progress bar
        for f in tqdm(concurrent.futures.as_completed(futures), total=len(target_layers), desc="Optimizing layers [MHE]"):
            f.result() # Check for exceptions

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    # Calculate MHE loss after optimization
    # optimized_mhe_loss


def mhe_optimized_init(model):
    """
    Sequential optimization for Single GPU to save memory and maximize throughput.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting MHE initialization on {device}...")

    # 1. Identify target layers (Linear layers, excluding lm_head)
    target_layers = []
    for module_name, module in model.named_modules():
        if "lm_head" in module_name:
            continue
        if isinstance(module, nn.Linear):
            target_layers.append((module_name, module))
            
            # Optional: Pre-normalize all layers in place (CPU or wherever they are)
            # with torch.no_grad():
            #      module.weight.div_(module.weight.norm(dim=1, keepdim=True))

    logger.info(f"Found {len(target_layers)} layers to optimize.")

    # 2. Sequential Processing Loop
    # We process one layer at a time to keep VRAM usage low.
    for i, (name, module) in enumerate(tqdm(target_layers, desc="Optimizing Layers")):
        
        # A. Move specific layer weight to GPU
        # We detach to ensure we don't drag the whole computation graph along
        weight_gpu = module.weight.detach().to(device).contiguous()
        
        # Double check normalization before optimization
        weight_gpu = weight_gpu / (weight_gpu.norm(dim=1, keepdim=True) + 1e-6)

        # B. Run the FAST vectorized optimization
        # (Make sure you are using the optimize_layer_mhe_fast from the previous step)
        optimized_weight = optimize_layer_mhe_fast(
            weight_gpu.to(torch.float32), # Ensure FP32 for precision
            n_steps=5000,
            lr=0.1,
            momentum=0.9,
            pd=40,
            pn=20,
            print_every=2000 # reduce spam
        )

        # C. Copy result back to original module
        # This allows the GPU memory for 'weight_gpu' and 'optimized_weight' to be freed
        with torch.no_grad():
            module.weight.data.copy_(optimized_weight.to(module.weight.device, dtype=module.weight.dtype))

        # D. cleanup
        del weight_gpu, optimized_weight
        
        # Aggressive cleanup every 10 layers to prevent fragmentation
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    logger.info("MHE Optimization Complete.")



def mhe_optimized_init_multi_gpu(model):
    """
    Distributes MHE optimization across ALL available GPUs on the node.
    Uses a Queue system to perfectly balance load (Dynamic Scheduling).
    """
    # 0. Check resources
    if not torch.cuda.is_available():
        print("No CUDA devices found. Falling back to sequential CPU.")
        mhe_optimized_init(model) 
        return

    num_gpus = torch.cuda.device_count()
    print(f"🚀 Starting Multi-GPU Optimization on {num_gpus} devices...")
    
    # Optional: Move model to CPU first to free up ALL VRAM for optimization
    # This prevents OOM if the model itself is large.
    model_device_initial = next(model.parameters()).device
    if model_device_initial.type != 'cpu':
        print("Moving model to CPU to free VRAM for workers...")
        model.cpu()

    # 1. Collect all target layers
    target_layers = []
    for name, module in model.named_modules():
        if "lm_head" in name: continue
        if isinstance(module, nn.Linear):
            # Pre-calculate norm on CPU to save GPU ops (optional)
            with torch.no_grad():
                module.weight.div_(module.weight.norm(dim=1, keepdim=True) + 1e-6)
            target_layers.append((name, module))
            
    print(f"Found {len(target_layers)} layers to optimize.")

    # 2. Fill the Queue
    layer_queue = queue.Queue()
    for item in target_layers:
        layer_queue.put(item)

    # Shared progress bar
    pbar = tqdm(total=len(target_layers), desc="Optimizing (Multi-GPU)")

    # 3. Define the Worker Function
    def gpu_worker(gpu_id):
        device = torch.device(f"cuda:{gpu_id}")
        
        # Keep processing until the queue is empty
        while True:
            try:
                # Get a layer from the queue (non-blocking)
                name, module = layer_queue.get_nowait()
            except queue.Empty:
                break # Nothing left to do, worker retires

            try:
                # A. Move weight to this worker's GPU
                # .detach() is crucial to break the graph
                weight_gpu = module.weight.detach().to(device).contiguous()
                
                # B. Optimize
                # We suppress print_every to avoid console spam from 4 threads
                optimized_weight = optimize_layer_mhe_fast(
                    weight_gpu.to(torch.float32), # Ensure FP32 for precision
                    n_steps=5000,
                    lr=0.1,
                    momentum=0.9,
                    pd=40,
                    pn=20,
                    print_every=2000 # reduce spam
                )

                # C. Copy result back to the module (usually on CPU)
                # We use a separate stream logic implicitly by moving data
                target_device = module.weight.device
                with torch.no_grad():
                    module.weight.data.copy_(optimized_weight.to(target_device))
                
                # D. Clean up GPU memory immediately
                del weight_gpu, optimized_weight
                
            except Exception as e:
                logger.error(f"Error on GPU {gpu_id} layer {name}: {e}")
            finally:
                # Mark task as done and update progress
                layer_queue.task_done()
                pbar.update(1)

    # 4. Launch Threads (1 per GPU)
    threads = []
    for i in range(num_gpus):
        t = threading.Thread(target=gpu_worker, args=(i,))
        t.daemon = True # Kills threads if main program crashes
        t.start()
        threads.append(t)

    # 5. Wait for all threads to finish
    for t in threads:
        t.join()
    
    pbar.close()
    
    # cleanup
    torch.cuda.empty_cache()
    print("✅ Multi-GPU Optimization Complete.")

    return model


def mhe_worker_process(model, worker_id, total_workers, save_dir):
    """
    Worker function for distributed MHE initialization via job scheduler (e.g. Condor).
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("Moving full model to CPU to free VRAM...")
    model.cpu()

    # Identify all target layers
    target_layers = []
    for module_name, module in model.named_modules():
        if "lm_head" in module_name:
            continue
        if isinstance(module, nn.Linear):
            target_layers.append((module_name, module))
    
    # Determine which layers this worker is responsible for
    my_layers = [item for i, item in enumerate(target_layers) if i % total_workers == worker_id]
    
    print(f"Worker {worker_id}/{total_workers} responsible for {len(my_layers)} layers.")
    
    if len(my_layers) == 0:
        print("No layers assigned to this worker.")
        return

    # Extract the weights we need and detach them to break the graph
    # We store them as (name, tensor_on_cpu)
    layers_to_process = []
    for name, module in my_layers:
        # Detach and move to CPU immediately
        weight_cpu = module.weight.detach().cpu().clone()
        layers_to_process.append((name, weight_cpu))

    # Now aggressively delete the model to free all resources
    print("Deleting full model to free memory...")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    for module_name, weight_cpu in layers_to_process:
        print(f"Processing layer: {module_name} with shape {weight_cpu.shape}")
        
        # 1. Move to GPU for optimization
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        # Move to device and normalize
        weight = weight_cpu.to(device).contiguous()
        weight = weight / weight.norm(dim=1, keepdim=True)
        
        # 2. Optimize (in FP32)
        optimized_fp32 = optimize_layer_mhe(weight.to(torch.float32))
        
        # 3. Save
        safe_name = module_name.replace(".", "_")
        save_path = os.path.join(save_dir, f"{safe_name}.pt")
        
        torch.save(optimized_fp32.cpu(), save_path)
        print(f"Saved optimized weights to {save_path}")
        
        # Cleanup loop variables
        del weight, optimized_fp32
        torch.cuda.empty_cache()
        gc.collect()


@torch.jit.script
def vectorized_thomson_loss(
    weight: torch.Tensor,
    n_projections: int,
    projection_dim: int,
    max_filt: int = 8912,
) -> torch.Tensor:
    """
    Vectorized Thomson loss using batch matrix multiplication.
    Args:
        weight: (out_features, in_features)
        n_projections: Number of random projections (pn)
        projection_dim: Dimension of projection space (pd)
    """
    n_filt, n_input = weight.shape
    device = weight.device
    dtype = weight.dtype


    # if n_filt > max_filt:
    #     idx = torch.randint(0, n_filt, (max_filt,), device=device, dtype=torch.int64)
    #     weight = weight.index_select(0, idx)
    #     n_filt = max_filt

    # 1. Generate all random projection matrices at once
    # Shape: (n_projections * projection_dim, n_input)
    # We essentially stack 'pn' projection matrices vertically
    projections = torch.randn(
        n_projections * projection_dim, n_input, 
        device=device, dtype=dtype
    )

    # 2. Project filters (Batch MatMul optimization)
    # W^T shape: (n_input, n_filt)
    # Result: (n_projections * projection_dim, n_filt)
    # Transposing weight once is faster
    w_t = weight.t()
    projected_flat = torch.mm(projections, w_t)

    # 3. Reshape to (n_projections, projection_dim, n_filt) for batch processing
    # We now have a batch of 'n_projections', each containing 'n_filt' vectors of size 'projection_dim'
    projected = projected_flat.view(n_projections, projection_dim, n_filt)
    
    # Transpose to (n_projections, n_filt, projection_dim) so vectors are in the last dim
    projected = projected.permute(0, 2, 1)

    # 4. Normalize vectors
    # Norm along the last dimension (projection_dim)
    # Shape: (n_projections, n_filt, 1)
    norms = torch.norm(projected, p=2, dim=2, keepdim=True) + 1e-6
    projected_normalized = projected / norms

    # 5. Create negative versions (MHE requirement)
    # Concatenate [v, -v] along the n_filt dimension
    # Shape: (n_projections, 2*n_filt, projection_dim)
    all_vectors = torch.cat([projected_normalized, -projected_normalized], dim=1)
    n_total = all_vectors.shape[1] # 2 * n_filt

    # 6. Compute Cosine Similarity via Batch Matrix Multiplication
    # (B, N, D) @ (B, D, N) -> (B, N, N)
    # resulting matrix contains cosine similarities for all pairs in the batch
    gram_matrix = torch.bmm(all_vectors, all_vectors.transpose(1, 2))

    # 7. Calculate Repulsion (Vectorized Riesz energy / Thomson problem)
    # Formula: sum( (2 - 2*cos_sim)^-1 )
    # We use a mask to ignore the diagonal (self-interaction) later
    
    # 2.0 - 2.0 * cos_sim
    distances = 2.0 - 2.0 * gram_matrix
    
    # Add epsilon to diagonal to prevent division by zero (self-interaction)
    # We construct a diagonal mask for the batch
    # eye = torch.eye(n_total, device=device, dtype=dtype).unsqueeze(0)
    # distances = distances + eye # Add 1.0 to diagonal (which is 0.0) -> avoids div/0
    distances.diagonal(dim1=-2, dim2=-1).add_(1.0)

    # Energy = 1 / distance
    energy = torch.reciprocal(distances)
    
    # 8. Summation
    # We want to sum the upper triangle (excluding diagonal)
    # Since the matrix is symmetric, Sum(Total) - Sum(Diagonal) = 2 * Sum(UpperTri)
    # Loss = Sum(UpperTri) / Count
    
    # Sum all elements
    sum_energy = energy.sum(dim=(1, 2))
    
    # Remove diagonal elements (which are 1/1 = 1.0 due to our epsilon trick)
    # The diagonal sum is exactly n_total
    actual_energy = sum_energy - n_total
    
    # Count of pairs: N * (N-1)
    # Note: The original code divides by N*(N-1)/2. 
    # Since we summed the full matrix (excluding diag), we have double the pairs.
    # So we divide by N*(N-1) to get the equivalent of summing upper triangle / count.
    cnt = n_total * (n_total - 1.0)
    
    batch_loss = actual_energy / cnt
    
    return batch_loss.mean()

def optimize_layer_mhe_fast(
    layer_weight,
    n_steps=2000,   # Reduced from 5000 (usually converges faster)
    lr=0.1, 
    momentum=0.9,
    pd=40,
    pn=20,
    print_every=500,
):
    device = layer_weight.device
    
    # Ensure weight is a leaf tensor with gradients
    weight = layer_weight.detach().clone().to(device)
    weight.requires_grad = True

    # Optimizer (SGD is fine, Adam usually converges in fewer steps for geometric problems)
    optimizer = torch.optim.SGD([weight], lr=lr, momentum=momentum)
    
    # Scheduler: Decay LR helps settle into the minimum
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # Pre-allocate reuse variables if needed, though JIT handles most
    # We do NOT calculate the exact MHE loss every step (it's O(N^2) and slow)
    for step in range(n_steps):
        # set_to_none=True is faster than zero_grad()
        optimizer.zero_grad(set_to_none=True)
        
        # Calculate loss (JIT compiled, vectorized)
        loss = vectorized_thomson_loss(weight, n_projections=pn, projection_dim=pd)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # if (step + 1) % print_every == 0 or step == 0:
        #     # Only print the approximate projection loss to save time
        #     # Calculating the REAL MHE loss (O(N^2)) is too slow to do in a loop
        #     print(f"Layer step [{step+1}/{n_steps}] Loss: {loss.item():.6f}")

    return weight.detach()



def optimize_layer_mhe(
    layer_weight,
    n_steps=5000, 
    lr=0.1, 
    momentum=0.9,
    print_every=500
):
    with torch.enable_grad():
        weight = nn.Parameter(layer_weight.clone().detach(), requires_grad=True).to(layer_weight.device)

        optimizer = torch.optim.SGD([weight], lr=lr, momentum=momentum)
        
        # Initial loss calculation
        with torch.no_grad():
            train_loss = thomson_random_project_loss(weight)
            init_mhe_loss = mhe_loss(weight)

        for step in range(n_steps):
            optimizer.zero_grad()
            train_loss = thomson_random_project_loss(weight)
            train_loss.backward()
            optimizer.step()
            
            if (step + 1) % 500 == 0:
                with torch.no_grad():
                    val_loss = mhe_loss(weight)
                    logger.info(f'Step [{step+1}/{n_steps}], '
                          f'Train Loss: {train_loss.item():.4f}, '
                          f'MHE Loss: {val_loss.item():.4f}')
            
            # train_losses.append(train_loss.item())

        final_mhe_loss = mhe_loss(weight)
        logger.info(f"Initial MHE loss: {init_mhe_loss:.8f}, Final MHE loss: {final_mhe_loss:.8f}")
    
    result = weight.detach().clone()
    del optimizer, weight, train_loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    
    return result
