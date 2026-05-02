def log_values(cost, distance, tardiness, grad_norms, epoch, batch_id, step,
               log_likelihood, reinforce_loss, bl_loss, tb_logger, opts, memory_stats=None):
    avg_cost = cost.mean().item()
    avg_distance = distance.mean().item()
    avg_tardiness = tardiness.mean().item()
    grad_norms, grad_norms_clipped = grad_norms

    # Log values to screen
    print('epoch: {}, train_batch_id: {}, avg_cost: {}'.format(epoch, batch_id, avg_cost))

    print('grad_norm: {}, clipped: {}'.format(grad_norms[0], grad_norms_clipped[0]))
    if memory_stats is not None:
        memory_msg = "memory"
        if 'gpu_alloc_mb' in memory_stats:
            memory_msg += ", gpu_alloc_mb: {:.2f}, gpu_reserved_mb: {:.2f}, gpu_peak_mb: {:.2f}".format(
                memory_stats['gpu_alloc_mb'],
                memory_stats['gpu_reserved_mb'],
                memory_stats['gpu_peak_mb']
            )
        if 'cpu_rss_mb' in memory_stats:
            memory_msg += ", cpu_rss_mb: {:.2f}".format(memory_stats['cpu_rss_mb'])
        print(memory_msg)

    # Log values to tensorboard
    if not opts.no_tensorboard:
        tb_logger.log_value('avg_cost', avg_cost, step)
        tb_logger.log_value('avg_distance', avg_distance, step)
        tb_logger.log_value('agv_tardiness', avg_tardiness, step)

        tb_logger.log_value('actor_loss', reinforce_loss.item(), step)
        tb_logger.log_value('nll', -log_likelihood.mean().item(), step)

        tb_logger.log_value('grad_norm', grad_norms[0], step)
        tb_logger.log_value('grad_norm_clipped', grad_norms_clipped[0], step)
        if memory_stats is not None:
            if 'gpu_alloc_mb' in memory_stats:
                tb_logger.log_value('gpu_mem_alloc_mb', memory_stats['gpu_alloc_mb'], step)
                tb_logger.log_value('gpu_mem_reserved_mb', memory_stats['gpu_reserved_mb'], step)
                tb_logger.log_value('gpu_mem_peak_mb', memory_stats['gpu_peak_mb'], step)
            if 'cpu_rss_mb' in memory_stats:
                tb_logger.log_value('cpu_rss_mb', memory_stats['cpu_rss_mb'], step)

        if opts.baseline == 'critic':
            tb_logger.log_value('critic_loss', bl_loss.item(), step)
            tb_logger.log_value('critic_grad_norm', grad_norms[1], step)
            tb_logger.log_value('critic_grad_norm_clipped', grad_norms_clipped[1], step)
