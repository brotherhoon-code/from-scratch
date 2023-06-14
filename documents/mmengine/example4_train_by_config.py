from mmengine.config import Config
from mmengine.runner import Runner
import example1_register # 필수 레지스트링 하는 모듈이 먼저 import 되어야 함.

config = Config.fromfile('./example3_config.py')
runner = Runner.from_cfg(config)
runner.train()