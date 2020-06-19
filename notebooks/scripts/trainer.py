from .pipeline import DatasetPipeline
import logging
import torch
logger = logging.getLogger('detectron2')

#uses whatever pycocotools is installed, make sure it is the TIL one
def do_test(cfg,model,dataset_name=None):
    from detectron2.evaluation import COCOEvaluator, inference_on_dataset
    from detectron2.data import build_detection_test_loader
    dataset_name = cfg.DATASETS.TEST[0] if dataset_name is None else dataset_name
    evaluator = COCOEvaluator(dataset_name,cfg,False,output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg,dataset_name)
    inference_on_dataset(model,val_loader,evaluator)
    return evaluator

def do_eval(cfg,model):
    from detectron2.data import build_detection_test_loader
    from tqdm import tqdm
    dataloader = build_detection_test_loader(cfg,cfg.DATASETS.TEST[0],mapper=DatasetPipeline(cfg,False))
    total_loss = 0
    logger.info("Calculating Validation Loss...")
    with torch.no_grad():
        for iteration,data in enumerate(tqdm(dataloader)):
            loss_dict = model(data)
            total_loss += sum(loss_dict.values())
    tqdm.write("\n")
    return total_loss/len(dataloader)

def do_train(cfg,model,model_context,resume=False,reset_val=False):
    from detectron2.utils.events import (
        CommonMetricPrinter,
        EventStorage,
        JSONWriter,
        TensorboardXWriter,
    )
    import detectron2.utils.comm as comm
    from detectron2.solver import build_lr_scheduler, build_optimizer
    from detectron2.data import build_detection_train_loader
    from detectron2.checkpoint import DetectionCheckpointer
    from .checkpointer import ValCheckpointer #eh, it gets the essence of early stopping & Im too lazy to make it actually early stop

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    saver = DetectionCheckpointer(model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler)
    meta = saver.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume) #loads the model weights & returns stored meta

    val_loss = meta.get('min_val_loss',meta.get('val_loss',999))
    if reset_val: val_loss = 999
    checkpointer = ValCheckpointer(saver,cfg.SAVE_EVERY,model_context,lambda: do_eval(cfg,model),val_loss)
    
    if 'EPOCHS' in cfg.SOLVER.keys(): cfg.SOLVER.MAX_ITER = cfg.SOLVER.EPOCHS*cfg.EPOCH_SIZE
    max_iter = cfg.SOLVER.MAX_ITER
    start_iter = meta.get("iteration",-1)+1 

    if resume: 
        logger.info(f"Resumed model: {meta.get('model_name','unknown')}")
        #override some configs from checkpoint in case you changed them
        scheduler.milestones = cfg.SOLVER.STEPS
        scheduler.gamma = cfg.SOLVER.GAMMA
        scheduler.base_lrs = [cfg.SOLVER.BASE_LR for lr in scheduler.base_lrs]
        scheduler.last_epoch = start_iter

    writers = [
        CommonMetricPrinter(max_iter),
        JSONWriter(f"{cfg.OUTPUT_DIR}/{model_context}-metrics.json"),
        TensorboardXWriter(cfg.OUTPUT_DIR),
    ]

    model.train() #set to training mode (PyTorch)
    dataloader = build_detection_train_loader(cfg,mapper=DatasetPipeline(cfg,True))
    logger.info(f"Training: Start Iter {start_iter}, End Iter {max_iter}")
    with EventStorage(start_iter) as storage:
        for iteration,data in zip(range(start_iter,max_iter),dataloader):
            try:
                iteration = iteration + 1
                storage.step()
                
                loss_dict = model(data)
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process(): storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
                scheduler.step()

                if iteration - start_iter > 5 and (iteration % cfg.PRINT_EVERY == 0 or iteration == max_iter):
                    for writer in writers: writer.write()
                checkpointer.step(iteration)
                
            except (Exception,KeyboardInterrupt) as e:
                logger.info("ERROR! Dumping current model...")
                checkpointer.save(f"{model_context}-{iteration}-interrupted",iteration=iteration,model_name=model_context,min_val_loss=checkpointer.min_loss)
                raise e
    checkpointer.save(f"{model_context}-{max_iter}-final",iteration=max_iter,model_name=model_context,min_val_loss=checkpointer.min_loss)
