from mindspore.train.callback import Callback
import moxing as mox


class UploadOutput(Callback):
    def __init__(self, train_dir, obs_train_url):
        self.train_dir = train_dir
        self.obs_train_url = obs_train_url

    def epoch_end(self, run_context):
        try:
            mox.file.copy_parallel(self.train_dir, self.obs_train_url)
            print("Successfully Upload {} to {}".format(self.train_dir, self.obs_train_url))
        except Exception as e:
            print('moxing upload {} to {} failed: '.format(self.train_dir, self.obs_train_url) + str(e))
        return
