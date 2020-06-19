'''
Logic for a validation loss based checkpointer, which detectron2 did not ship with.
'''
from fvcore.common.checkpoint import PeriodicCheckpointer
from detectron2.engine import HookBase
import logging

#TODO: add early stopping (not necessary; only saves on val drop, and im not using a rate-limited service)
class ValCheckpointer(PeriodicCheckpointer, HookBase):
    def __init__(self,checkpointer,save_every,model_context,eval_func,min_val_loss=999):
        self.checkpointer = checkpointer
        self.period = save_every
        self.model_name = model_context
        self.evaluator = eval_func
        self.min_loss = min_val_loss
        self.logger = logging.getLogger('detectron2')
        if self.min_loss == None: self.min_val_loss = 999

    def step(self,iters,**kwargs):
        if (iters+1)%self.period != 0: return
        
        willSave = False
        val_loss = self.evaluator()
        if val_loss < self.min_loss: willSave = True

        saveMsg = "saving model." if willSave else "did not save model."
        self.logger.info(f"Val Loss: {val_loss} @ Iteration {iters}, Min Val Loss: {self.min_loss}, {saveMsg}")
        if not willSave: return
        self.min_loss = val_loss

        meta = {}
        meta['iteration'] = iters
        meta['model_name'] = self.model_name
        meta['val_loss'] = val_loss
        meta['min_val_loss'] = val_loss
        meta.update(kwargs)

        self.checkpointer.save(f"{self.model_name}-{iters}-best_val", **meta)
