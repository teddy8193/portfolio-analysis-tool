import logging

logger = logging.getLogger('PortfolioTool')
logger.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s - %(message)s')

# Stream Handler
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
