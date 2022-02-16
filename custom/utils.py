import os
import shutil

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback

class saveCallback(CheckpointCallback):
    def __init__(self, save_freq, name_prefix="ckpt", verbose=0):
        super(saveCallback, self).__init__(save_freq, save_path=None, name_prefix=name_prefix, verbose=verbose)

    def _on_step(self) -> bool:
        if self.save_path is None:
            self.save_path = os.path.join(self.model._logger.dir, "ckpt")
            if not os.path.isdir(self.save_path):
                os.mkdir(self.save_path)

        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.verbose > 1:
                print(f"Saving model checkpoint to {path}")
        return True

class configCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(configCallback, self).__init__(verbose)
        self.save_path = None

    def _on_training_start(self) -> None:
        self.save_path = os.path.join(self.model._logger.dir, 'config')
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        shutil.copyfile('config/train.yaml', os.path.join(self.save_path,"train.yaml"))

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass