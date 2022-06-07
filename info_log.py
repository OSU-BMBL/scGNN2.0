import logging
# A Unix specific library for Python
# import resource

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

def print(info, include_mem=False):
    logging.info(info)

    if include_mem:
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        logging.info(f'Memory consumption (Kb): {mem}')

def interval_print(info, epoch=0, total_epoch=0, interval=20):
    epoch += 1
    if epoch == 1 or epoch == total_epoch or epoch % interval == 0:
        logging.info(f'--------{info}')