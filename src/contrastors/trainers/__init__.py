from .base import *
from .glue import *
from .mlm import *
from .text_text import *

TRAINER_REGISTRY = {
    "mlm": MLMTrainer,
    "glue": GlueTrainer,
    "encoder": TextTextTrainer,
}
