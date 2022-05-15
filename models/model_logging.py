import logging

logger = logging.Logger("log")
consoleHandler = logging.StreamHandler()
logFormatter = logging.Formatter("%(asctime)s %(message)s")
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
